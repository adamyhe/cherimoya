"""Tests of the fast DeepLIFT/SHAP engine against tangermeme's
`deep_lift_shap`, on the CPU.

Both sides get the same 13 sequences (12 random, 1 AT-rich) and the same
injected references, K = 5 per sequence, of four kinds: dinucleotide
shuffles, random one-hots, r = x, where every rule takes the plain
gradient, and x with 3 mutated positions, where the rescale rule's inputs
differ by less than 1e-6 at some position of every GELU and it takes the
gradient there. The oracle is `deep_lift_shap` as `cherimoya attribute`
calls it, on the CPU, at batch_size 7, which splits a sequence's pairs
across batches. The engine runs in steps of 8 sequences, so 13 sequences
fill one step and pad the second with 3 copies of the last one.

* Hypothetical attributions, per sequence: <= 1e-9 relative L2 in float64
  and 1e-5 in float32.
* Convergence deltas in float64: |delta_ours - delta_tangermeme| <= 1e-9 x
  max(1, |y_x - y_r|), with tangermeme's deltas computed from its
  multipliers.
* Steps of 3 and 8 sequences give the results of steps of 1 to 1e-12.

The engine's CPU forward is the model's CPU path, which normalizes in
float32 even in a float64 model, as the oracle's forward does.
"""

import functools
import re
import warnings

import pytest
import torch

from cherimoya import ControlWrapper
from cherimoya import _deep_lift_shap_engine as fdls
from cherimoya import _pisa_engine as fpisa
from cherimoya.deep_lift_shap import _make_wrapper

from . import deep_lift_shap_engine_helpers as tiny


K = 5
F64 = torch.float64
N_SEQUENCES = tiny.N_RANDOM + tiny.N_AT_RICH

HYP_TOL = {torch.float32: 1e-5, F64: 1e-9}

# The profile heads' rules divide changes in the logits by changes in the
# logits, so in float32 the rounding of a logit that barely moves is
# amplified, on either path. The engine computes those rules in float64, so
# in float32 both paths are held to the float64 oracle: the engine at most
# `PROFILE_F32_RATIO` times as far from it as tangermeme's float32 path,
# plus `PROFILE_F32_SLACK` (at most 1.03 times and up to 30 times closer on
# these cases), and within `PROFILE_F32_MAX` (1.0e-4 at most).
PROFILE_F64_TOL = 1e-9
PROFILE_F32_RATIO, PROFILE_F32_SLACK, PROFILE_F32_MAX = 1.25, 1e-6, 2e-4
PROFILE_OUTPUTS = ("profile", "negative_entropy")
DELTA_FP64 = 1e-9
Y_FP64 = 1e-12
STEP_TOL = 1e-12
RULES_MIN = 0.1

# Every kind of reference on the small configs A and B; shuffled ones, the
# default path's, on C, whose float64 oracle takes seconds per call.
PARITY_CASES = [(config, dtype, kind) for config in ("A", "B")
	for dtype, kind in ((torch.float32, "shuffled"), (torch.float32, "random"),
	(F64, "shuffled"), (F64, "random"), (F64, "identical"), (F64, "mutated"))
	] + [("C", torch.float32, "shuffled"), ("C", F64, "shuffled")]


@functools.cache
def _case(config, kind):
	"""A config's 13 sequences, int8 (13, 4, L), and K references of one
	kind, float32 (13, K, 4, L)."""

	X = tiny.sequences(config)
	if kind == "shuffled":
		return X, tiny.shuffled_references(X, K)

	if kind == "random":
		return X, tiny.random_references(X, K, seed=3)

	if kind == "identical":
		return X, tiny.identical_references(X, K)

	if kind == "mutated":
		return X, tiny.mutated_references(X, K, seed=5)

	raise ValueError(kind)


_GROUP = object()
"""`_engine`'s and `_wrapped`'s default group: the config's."""


def _engine(config, dtype, output="logcounts", group=_GROUP, **kwargs):
	"""An Engine on a fresh tiny model, attributing the config's group."""

	model = tiny.tiny_model(config, dtype=dtype)
	group = tiny.CONFIGS[config].group if group is _GROUP else group
	return fdls.Engine(model, output=output, group=group, **kwargs)


@functools.cache
def _ours(config, dtype, kind, step=None, output="logcounts", group=_GROUP):
	X, R = _case(config, kind)
	return _engine(config, dtype, output, group, seqs_per_step=step).run(X, R)


def _wrapped(config, dtype, output="logcounts", group=_GROUP):
	model = tiny.tiny_model(config, dtype=dtype)
	group = tiny.CONFIGS[config].group if group is _GROUP else group
	return _make_wrapper(model, output, group)


def _dy(result):
	"""|y_x - y_r| per pair, (N, K), in float64."""

	return (result.y_x[:, None] - result.y_r).double().abs()


##


@pytest.mark.parametrize("config, dtype, kind", PARITY_CASES)
def test_attributions_match_tangermeme(config, dtype, kind):
	X, R = _case(config, kind)
	ours = _ours(config, dtype, kind)
	theirs = tiny.oracle(_wrapped(config, dtype), X, R, dtype)
	assert ours.attr.shape == theirs.shape == X.shape
	assert ours.attr.dtype == ours.deltas.dtype == ours.y_x.dtype == dtype
	assert ours.deltas.shape == ours.y_r.shape == (N_SEQUENCES, K)
	assert float(tiny.rows_rel_l2(ours.attr, theirs).max()) <= HYP_TOL[dtype]


# The profile heads on both configs, for the config's group and for every
# group together (None), with every kind of reference, in float32 and
# float64.
PROFILE_CASES = [(config, dtype, kind, output, group)
	for config in ("A", "B") for output in PROFILE_OUTPUTS
	for group in (tiny.CONFIGS[config].group, None)
	for dtype in (torch.float32, F64)
	for kind in ("shuffled", "random", "identical", "mutated")]


@pytest.mark.parametrize("config, dtype, kind, output, group", PROFILE_CASES)
def test_profile_attributions_match_tangermeme(config, dtype, kind, output,
	group):
	X, R = _case(config, kind)
	ours = _ours(config, dtype, kind, output=output, group=group)
	truth = tiny.oracle(_wrapped(config, F64, output, group), X, R, F64)
	assert ours.attr.shape == truth.shape == X.shape
	assert ours.attr.dtype == ours.deltas.dtype == ours.y_x.dtype == dtype
	assert ours.deltas.shape == ours.y_r.shape == (N_SEQUENCES, K)

	error = float(tiny.rows_rel_l2(ours.attr, truth).max())
	if dtype == F64:
		assert error <= PROFILE_F64_TOL
		return

	theirs = tiny.oracle(_wrapped(config, dtype, output, group), X, R, dtype)
	their_error = float(tiny.rows_rel_l2(theirs, truth).max())
	assert error <= PROFILE_F32_RATIO * their_error + PROFILE_F32_SLACK
	assert error <= PROFILE_F32_MAX


@pytest.mark.parametrize("output", PROFILE_OUTPUTS)
@pytest.mark.parametrize("config", ["A", "B"])
def test_profile_attributions_are_head_weighted_pisa(config, output):
	"""With one reference per sequence, a profile head's attributions are
	the PISA engine's, one per attributed channel and output position,
	weighted by the head's multipliers at those logits, in float64.

	Below the logits every rule's multipliers are fixed by the forward,
	so the backward is linear in the gradient it starts from. PISA starts
	it from the profile convolution's taps at one logit, and this engine
	from conv1d_input of the head's multipliers m, which is the sum of
	those starting gradients times m. So for each pair the attributions
	are sum_j m_j * PISA_j. m depends on the reference, so the identity
	holds per pair, not for attributions averaged over several references,
	and K = 1 makes the two the same. Config B's group has two channels,
	so the sum also runs over channels, flattened channel by channel."""

	X = tiny.sequences(config)[:3]
	R = tiny.random_references(X, 1, seed=3)
	model = tiny.tiny_model(config, dtype=F64)
	group = tiny.CONFIGS[config].group
	start = sum(model.signal_groups[:group])
	end = start + model.signal_groups[group]

	ours = fdls.Engine(model, output=output, group=group).run(X, R)
	pisa = torch.stack([fpisa.Engine(model, channel=c, n_shuffles=1)
		.run(X, R).attr for c in range(start, end)], dim=1)

	with torch.no_grad():
		logits = ControlWrapper(model)(torch.cat([X, R[:, 0]]).to(F64))[0]
	_, m = fdls.profile_head_rule(logits[:, start:end].flatten(1), len(X), 1,
		output)
	expected = torch.einsum("ncj,ncjal->nal", m.view(pisa.shape[:3]), pisa)

	assert ours.attr.shape == expected.shape == X.shape
	assert float(tiny.rows_rel_l2(ours.attr, expected).max()) <= PROFILE_F64_TOL


@pytest.mark.parametrize("output", ["logcounts", *PROFILE_OUTPUTS])
@pytest.mark.parametrize("config", ["A", "B"])
def test_outputs_and_deltas_match_tangermeme(config, output):
	"""In float64: the outputs are the module's, the deltas are tangermeme's
	expression over its own multipliers, and those multipliers are at least
	10% from the plain gradient for every pair, so the tests here exercise
	the rules."""

	X, R = _case(config, "shuffled")
	wrapped = _wrapped(config, F64, output)
	ours = _ours(config, F64, "shuffled", output=output)
	m = tiny.oracle(wrapped, X, R, F64, raw_outputs=True)

	x, r = X.to(F64), R.to(F64)
	with torch.no_grad():
		y_x = wrapped(x)[:, 0]
		y_r = wrapped(r.flatten(0, 1))[:, 0].view(N_SEQUENCES, K)

	assert float(((ours.y_x - y_x).abs() / y_x.abs().clamp_min(1)).max()) \
		<= Y_FP64
	assert float(((ours.y_r - y_r).abs() / y_r.abs().clamp_min(1)).max()) \
		<= Y_FP64

	deltas = ((y_x[:, None] - y_r) - ((x[:, None] - r) * m).sum(dim=(-2,
		-1))).abs()
	err = (ours.deltas - deltas).abs() / _dy(ours).clamp_min(1)
	assert float(err.max()) <= DELTA_FP64

	with torch.enable_grad():
		leaf = x.clone().requires_grad_()
		gradient, = torch.autograd.grad(wrapped(leaf)[:, 0].sum(), leaf)

	rel = tiny.rows_rel_l2(m.flatten(0, 1), gradient.repeat_interleave(K, 0))
	assert float(rel.min()) >= RULES_MIN


@pytest.mark.parametrize("output", ["logcounts", "profile"])
@pytest.mark.parametrize("config", ["A", "B"])
def test_results_do_not_depend_on_the_step_size(config, output):
	"""Steps of 3 and 8 sequences against steps of 1, for 13 sequences, the
	last step padded with copies of the last sequence. Rows are independent
	in every operation, so only reassociation can differ."""

	base = _ours(config, F64, "shuffled", 1, output)
	for step in (3, 8):
		ours = _ours(config, F64, "shuffled", step, output)
		y_ref = torch.cat([base.y_x[:, None], base.y_r], dim=1).double()
		y = torch.cat([ours.y_x[:, None], ours.y_r], dim=1).double()
		errs = {
			"attr": float(tiny.rows_rel_l2(ours.attr, base.attr).max()),
			"deltas": float(((ours.deltas - base.deltas).abs()
				/ _dy(base).clamp_min(1)).max()),
			"y": float(((y - y_ref).abs() / y_ref.abs().clamp_min(1)).max()),
		}

		assert max(errs.values()) <= STEP_TOL, (step, errs)


def test_run_warns_once_per_check(monkeypatch):
	"""One RuntimeWarning per run when a delta exceeds the threshold, none
	for a threshold of None, and one when the mean delta exceeds
	RELATIVE_DELTA_MAX of the mean output difference."""

	X, R = _case("A", "shuffled")

	def caught(**kwargs):
		with warnings.catch_warnings(record=True) as records:
			warnings.simplefilter("always")
			result = _engine("A", torch.float32, **kwargs).run(X, R)

		return result, [str(w.message) for w in records
			if issubclass(w.category, RuntimeWarning)]

	result, messages = caught(warning_threshold=0.0)
	assert len(messages) == 1
	assert re.fullmatch(r"Convergence deltas too high: \d+ of 65 pairs > 0, "
		r"the largest \S+", messages[0])

	assert caught(warning_threshold=None)[1] == []
	assert caught(warning_threshold=float(result.deltas.max()))[1] == []

	relative = float(result.deltas.double().mean() / _dy(result).mean())
	assert 0 < relative < fdls.RELATIVE_DELTA_MAX
	monkeypatch.setattr(fdls, "RELATIVE_DELTA_MAX", relative / 2)
	messages = caught(warning_threshold=None)[1]
	assert len(messages) == 1
	assert re.fullmatch(r"Convergence deltas are \S+% of the output "
		r"differences over 65 pairs, above \S+%: the attributions do not sum "
		r"to the change in the output", messages[0])

class AttnResMixer:
    """x_{l+1} = x_l + y_l + sum_i alpha[l][i] * y_i (y_i = layer OUTPUT)."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.alpha = [[0.0] * l for l in range(n_layers)]

    def set(self, layer: int, weights: list) -> None:
        if len(weights) != layer:
            raise ValueError(f"layer {layer} expects {layer} weights")
        self.alpha[layer] = [float(w) for w in weights]

    def correction(self, layer: int, prev_outputs: list) -> list:
        if layer == 0 or not any(self.alpha[layer]):
            return [0.0] * len(prev_outputs[0])
        acc = [0.0] * len(prev_outputs[0])
        for a, y in zip(self.alpha[layer], prev_outputs):
            if a != 0.0:
                for j in range(len(acc)):
                    acc[j] += a * y[j]
        return acc

    def fingerprint(self) -> tuple:
        return (self.n_layers, tuple(tuple(r) for r in self.alpha))


def _vadd(a, b):
    return [x + y for x, y in zip(a, b)]


def run_stack(layer_fns: list, x0: list, mixer=None) -> list:
    x, ys = list(x0), []
    for l, f in enumerate(layer_fns):
        y = f(x)
        corr = mixer.correction(l, ys) if (mixer is not None and l > 0) \
            else [0.0] * len(y)
        x = _vadd(_vadd(x, y), corr)
        ys.append(y)
    return x


def migration_gate(layer_fns: list, x0: list, n_layers: int) -> None:
    """Zero-init mixer must reproduce the vanilla stack BITWISE."""
    if run_stack(layer_fns, x0) != run_stack(
            layer_fns, x0, mixer=AttnResMixer(n_layers)):
        raise AssertionError(
            "ATTNRES MIGRATION GATE VIOLATED: zero-init mixer changed the "
            "output — alpha path is not identity at init; this is a bug")


def determinism_gate(layer_fns: list, x0: list, mixer: AttnResMixer) -> None:
    a = run_stack(layer_fns, x0, mixer=mixer)
    b = run_stack(layer_fns, x0, mixer=mixer)
    if a != b:
        raise AssertionError(
            "ATTNRES DETERMINISM GATE VIOLATED: same inputs, different "
            "outputs — a hidden source of nondeterminism exists")

import matplotlib.pyplot as plt
import random

# Define IFS rules: (a, b, d, e, c, f, probability)
ifs_rules = [
    (.824074, .281482, -.212346, .864198, -1.882290, -0.110607, .787473),
    (.088272, .520988, -.463889, -.377778, 0.785360, 8.095795, .212527)
]

def apply_transform(x, y, rule):
    a, b, d, e, c, f, _ = rule
    # a, b, c, d, e, f, _ = rule
    x_new = a * x + b * y + e
    y_new = c * x + d * y + f
    return x_new, y_new

def choose_rule(rules):
    r = random.random()
    s = 0
    for rule in rules:
        s += rule[6]
        if r <= s:
            return rule
    return rules[-1]

def generate_ifs(rules, n_points=1_000_000, discard=0):
    x, y = 0, 0
    points_x, points_y = [], []
    for i in range(n_points + discard):
        rule = choose_rule(rules)
        x, y = apply_transform(x, y, rule)
        if i >= discard:
            points_x.append(x)
            points_y.append(y)
    return points_x, points_y

# Generate and plot
x_vals, y_vals = generate_ifs(ifs_rules)
plt.figure(figsize=(10, 10))
plt.scatter(x_vals, y_vals, s=0.1, color='green')
plt.axis('off')
plt.title("IFS Fractal")
plt.show()

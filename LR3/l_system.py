import turtle

# L-system settings
axiom = "X"
rules = {
    "X": "+FF-YFF+FF--FFF|X|F--YFFFYFFF|",
    "Y": "-FF+XFF-FF++FFF|Y|F++XFFFXFFF|",
    "F": "GG",
    "G": "GG"
}
angle = 60

def apply_rules(axiom, rules, iterations):
    result = axiom
    for _ in range(iterations):
        new_result = ""
        for char in result:
            new_result += rules.get(char, char)  # default to char if no rule
        result = new_result
    return result

import turtle
import math

def simulate_path(instructions, angle=60, step=5):
    x, y = 0, 0
    heading = 0
    positions = [(x, y)]

    for cmd in instructions:
        if cmd in "FG":
            rad = math.radians(heading)
            x += step * math.cos(rad)
            y += step * math.sin(rad)
            positions.append((x, y))
        elif cmd == "+":
            heading += angle
        elif cmd == "-":
            heading -= angle
        elif cmd == "|":
            heading += 180

    return positions

def get_bounds(positions):
    xs, ys = zip(*positions)
    return min(xs), max(xs), min(ys), max(ys)

def draw_lsystem_centered(instructions, angle=60, step=5):
    positions = simulate_path(instructions, angle, step)
    min_x, max_x, min_y, max_y = get_bounds(positions)

    offset_x = -(min_x + max_x) / 2
    offset_y = -(min_y + max_y) / 2

    turtle.speed(0)
    turtle.hideturtle()
    turtle.tracer(0, 0)
    turtle.penup()
    turtle.goto(offset_x, offset_y)
    turtle.setheading(0)
    turtle.pendown()

    for cmd in instructions:
        if cmd in "FG":
            turtle.forward(step)
        elif cmd == "+":
            turtle.left(angle)
        elif cmd == "-":
            turtle.right(angle)
        elif cmd == "|":
            turtle.setheading((turtle.heading() + 180) % 360)

    turtle.update()
    turtle.done()


# Build and draw
instructions = apply_rules(axiom, rules, iterations=6)  # try 3 or 4 first
draw_lsystem_centered(instructions, angle)

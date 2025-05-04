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

def draw_lsystem(instructions, angle, step=5):
    turtle.speed(0)
    turtle.hideturtle()
    turtle.penup()
    turtle.goto(-200, -200)
    turtle.setheading(0)
    turtle.pendown()

    stack = []

    for cmd in instructions:
        if cmd in "FG":
            turtle.forward(step)
        elif cmd == "+":
            turtle.left(angle)
        elif cmd == "-":
            turtle.right(angle)
        elif cmd == "|":
            # optional: reflect direction
            turtle.setheading((turtle.heading() + 180) % 360)
        # You can add '[' and ']' for pushing/popping stack if needed

    turtle.done()

def draw_lsystem_fast(instructions, angle=60, step=5):
    turtle.speed(0)
    turtle.goto(0, 0)
    turtle.hideturtle()
    turtle.tracer(0, 0)  # Disable animation
    for cmd in instructions:
        if cmd in "FG":
            turtle.forward(step)
        elif cmd == "+":
            turtle.left(angle)
        elif cmd == "-":
            turtle.right(angle)
        elif cmd == "|":
            turtle.setheading((turtle.heading() + 180) % 360)
    turtle.update()  # Render all at once
    turtle.done()

# Build and draw
instructions = apply_rules(axiom, rules, iterations=5)  # try 3 or 4 first
draw_lsystem_fast(instructions, angle)

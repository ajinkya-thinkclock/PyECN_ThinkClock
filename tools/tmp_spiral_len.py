import math

nstack_real = 22

delta_el = 0.00017718
delta_al = 1.633e-05
delta_cu = 2.7e-05
delta_core = 0.00192

a0 = (2 * delta_el + delta_cu + delta_al) / (2 * math.pi)
b0 = delta_el / 2 + delta_al + delta_core
theta = nstack_real * 2 * math.pi


def spiral_len(a0_local, b0_local, theta_local):
    return (
        (a0_local * theta_local + b0_local)
        * math.sqrt(
            a0_local * a0_local
            + b0_local * b0_local
            + 2 * a0_local * b0_local * theta_local
            + a0_local * a0_local * theta_local * theta_local
        )
        / 2
        / a0_local
        + a0_local
        / 2
        * math.log(
            theta_local
            + b0_local / a0_local
            + math.sqrt((theta_local + b0_local / a0_local) ** 2 + 1)
        )
        - math.sqrt(a0_local * a0_local + b0_local * b0_local)
        * b0_local
        / 2
        / a0_local
        - a0_local
        / 2
        * math.log(b0_local / a0_local + math.sqrt((b0_local / a0_local) ** 2 + 1))
    )


L_s = spiral_len(a0, b0, theta)
L_l = spiral_len(a0, b0 + delta_el + delta_cu, theta)

print("Spiral_Sep_s_real", L_s)
print("Spiral_Sep_l_real", L_l)
print("Sum", L_s + L_l)

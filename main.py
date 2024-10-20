import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from matplotlib.animation import FuncAnimation
# mass
m1 = 1
m2 = 1
m3 = 1
# initial position
pos_1 = [1, 0, 1]
pos_2 = [1, 1, 0]
pos_3 = [0, 1, 1]
# initial velocity
v1 = [0, 0, -1]
v2 = [0, 0, 1]
v3 = [0, 0, -0.6]

initial_conditions = np.array([pos_1, pos_2, pos_3, v1, v2, v3]).ravel()
def solve_odes(t, S, m1, m2, m3):
    p1, p2, p3 = S[0:3], S[3: 6], S[6:9]
    f1, f2, f3 = S[9:12], S[12:15], S[15:18]
    df1_dt = (m2 * (p2 - p1) / np.linalg.norm(p2 - p1) ** 3) + (m3 * (p3 - p1) / np.linalg.norm(p3 - p1) ** 3)
    df2_dt = (m1 * (p1 - p2) / np.linalg.norm(p1 - p2) ** 3) + (m3 * (p3 - p2) / np.linalg.norm(p3 - p2) ** 3)
    df3_dt = (m2 * (p1 - p3) / np.linalg.norm(p1 - p3) ** 3) + (m3 * (p2 - p3) / np.linalg.norm(p2 - p3) ** 3)

    return np.array([f1, f2, f3, df1_dt, df2_dt, df3_dt]).ravel()


time_s, time_e = 0, 10
timePoints = np.linspace(time_s, time_e, 1001)
solution = solve_ivp(
    fun=solve_odes,
    t_span=(time_s, time_e),
    y0=initial_conditions,
    t_eval=timePoints,
    args=(m1, m2, m3)
)
# Storing the calculated values in individual values
t_solution = solution.t
p1x = solution.y[0]
p2x = solution.y[1]
p3x = solution.y[2]

p1y = solution.y[3]
p2y = solution.y[4]
p3y = solution.y[5]

p1z = solution.y[6]
p2z = solution.y[7]
p3z = solution.y[8]



# Plotting the graph
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
body_1, = ax.plot(p1x, p1y, p1z, 'green', label='Body 1', linewidth=1)
body_2, = ax.plot(p2x, p2y, p2z, 'red', label='Body 2', linewidth=1)
body_3, = ax.plot(p3x, p3y, p3z, 'blue', label='Body 3', linewidth=1)

body_1dot, = ax.plot([p1x[-1]], [p1y[-1]], [p1z[-1]], 'o', color='green', markersize=5)
body_2dot, = ax.plot([p2x[-1]], [p2y[-1]], [p2z[-1]], 'o', color='red', markersize=5)
body_3dot, = ax.plot([p3x[-1]], [p3y[-1]], [p3z[-1]], 'o', color='blue', markersize=5)
def update(frame):
    x_current1 = p1x[0:frame+1]
    y_current1 = p1y[0:frame+1]
    z_current1 = p1z[0:frame+1]

    x_current2 = p2x[0:frame+1]
    y_current2 = p2y[0:frame+1]
    z_current2 = p2z[0:frame+1]

    x_current3 = p3x[0:frame + 1]
    y_current3 = p3y[0:frame + 1]
    z_current3 = p3z[0:frame + 1]


    body_1.set_data(x_current1, y_current1)
    body_1.set_3d_properties(z_current1)

    body_1dot.set_data([x_current1[-1]], [y_current1[-1]])
    body_1dot.set_3d_properties([z_current1[-1]])

    body_2.set_data(x_current2, y_current2)
    body_2.set_3d_properties(z_current2)

    body_2dot.set_data([x_current2[-1]], [y_current2[-1]])
    body_2dot.set_3d_properties([z_current2[-1]])

    body_3.set_data(x_current3, y_current3)
    body_3.set_3d_properties(z_current3)

    body_3dot.set_data([x_current3[-1]], [y_current3[-1]])
    body_3dot.set_3d_properties([z_current3[-1]])

    return body_1dot, body_2dot, body_3dot, body_1, body_2, body_3

animation = FuncAnimation(fig, update, frames=range(0, len(timePoints), 1), interval=5, blit=True)
plt.show()



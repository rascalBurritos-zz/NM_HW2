#%%
"""

Given a system of masses and their location relative to each other

Plot the gravitational potential as a heatmap

Plot the Gravitational Field as a quiver plot

"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import reduce
#%%
@dataclass
class Position:
  x : float # in 10^7 km
  y : float # in 10^7 km
  def distance_to(self, pos):
    return np.sqrt((self.x-pos.x)**2 +(self.y - pos.y)**2)

@dataclass
class MassiveObject:
  pos : Position 
  mass : float # in solar masses

#%%
def gravity_potential(target_point : Position, massive_objects, true_to_reality=False ):
  """
  V(x) = sum(i=1->n, -G*m_i/|x-x_i|)
  """
  #units of this G = 10^10 [m^3 kg^-1 s^-2] 
  G = (6.67430e-11*10e-10) if true_to_reality else 1.0
  def pot_accumulator(accum_pot : float, cur_mass_obj : MassiveObject):
    cur_mass_obj.mass = cur_mass_obj.mass * 2e30
    return accum_pot + -G *cur_mass_obj.mass/(cur_mass_obj.pos.distance_to(target_point))
  return reduce(pot_accumulator,massive_objects,0)

def potential(target_point: Position, true_to_reality=False):
  """
  V(x) = sum(i=1->n, -G*m_i/|x-x_i|)
  """
  #units of this G = 10^10 [m^3 kg^-1 s^-2] 
  G = (6.67430e-11*10e-10) if true_to_reality else 1.0
  mo1 = MassiveObject(Position(0.0,0.0), 1)
  mo2 = MassiveObject(Position(3,5), 2.5)
  mo3 = MassiveObject(Position(4,6), 0.5)
  return gravity_potential(target_point,(mo1,mo2,mo3),true_to_reality=true_to_reality)
# %%
# Sample Parameters
x_sample_params = (-10, 10, 100) # min, max , n
y_sample_params = (-10, 10, 100) # min, max , n
x_coordinates, x_step = np.linspace(*x_sample_params, retstep=True) 
y_coordinates, y_step = np.linspace(*y_sample_params, retstep=True)  
xx, yy = np.meshgrid(x_coordinates, y_coordinates, indexing='ij')
# %%
sampled_potential = np.empty(xx.shape)
for i in range(xx.shape[0]):
  for j in range(xx.shape[1]):
    sample_position = Position(xx[i,j], yy[i,j])
    sampled_potential[i,j] = potential(sample_position)

sampled_field =  -np.array(np.gradient(sampled_potential,x_step,y_step))
mag_sampled_field = np.sqrt((sampled_field[0]**2+sampled_field[1]**2))
normalized_sampled_field = sampled_field/mag_sampled_field
# %%
contour_levels = 250
plt.style.use('ggplot')
fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
ax1.contour(xx,yy,sampled_potential, contour_levels, cmap='magma')
ax1.contourf(xx,yy,sampled_potential, contour_levels, cmap='magma')
ax1.set_aspect('equal')

ax2.quiver(xx,yy,normalized_sampled_field[0],normalized_sampled_field[1], mag_sampled_field)
ax2.set_aspect('equal')
ax2.xaxis.set_ticks([])
ax2.yaxis.set_ticks([])
plt.savefig('output.svg')

# %%
# START - Taken from in class jupyter notebook
def partial_derivative(f, p, i, h):        
        pfp = np.array([p_j + (h if j == i else 0) for j, p_j in enumerate(p)])
        pfm = np.array([p_j - (h if j == i else 0) for j, p_j in enumerate(p)])
        return (f(pfp) - f(pfm)) / (2*h)
def estimate_grad(f, p, h):
    return np.array([partial_derivative(f, p, i, h) for i, _ in enumerate(p)])

# END - Taken from in class jupyter notebook
sci = np.format_float_scientific
# Determine output halfwave between massive objects
print(sci(potential(Position(1.5,2.5),true_to_reality=True))) #[Joule kg^-1]
print(*[sci(-i) for i in estimate_grad(lambda p:potential(Position(*p),true_to_reality=True), (1.5, 2.5),h=0.001)])

# %%
sun_pot = gravity_potential(
  Position(0,11.5),
  [MassiveObject(Position(0,0),1)],
  true_to_reality=True)
print(sci(sun_pot))
sun_field = estimate_grad(lambda p:
gravity_potential(Position(*p), [MassiveObject(Position(0,0),1)],true_to_reality=True), (0, 11.5),h=0.001)
print(*[sci(-i) for i in sun_field])
# %%
#Q2

class MaxIterationsExceeded(Exception):
  pass


def gradient_descent(f, pos : np.array, learning_rate = 0.01, max_iters=10000, threshold = 1e-5):
  current_pos = pos
  for _ in range(max_iters):
    grad_val = estimate_grad(f,current_pos, h=0.001 )
    next_pos = pos - learning_rate*grad_val
    mag_change = np.linalg.norm(current_pos-next_pos)
    current_pos = next_pos
    print(current_pos)
    if mag_change < threshold:
      return current_pos
  raise MaxIterationsExceeded()


# Starting Question 2
def himmel_func(pos):
  x = pos[0]
  y = pos[1]
  return (x**2+y-11)**2+(x+(y**2)-7)**2 

# %%
gradient_descent(himmel_func,np.array([0.0,0.0]))

# %%
# estimate_grad(himmel_func,np.array([0,0]), h=0.001)
# next_pos = np.array[0,0]
# %%

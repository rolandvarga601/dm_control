from absl import app
from dm_control import manipulation, mujoco
from dm_control.entities.manipulators import kinova
from dm_control import composer
from dm_control import mjcf

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums

import numpy as np

from dm_control import viewer

import PIL.Image


class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()



# def sample_random_action():
#   return env.random_state.uniform(
#       low=action_spec.minimum,
#       high=action_spec.maximum,
#   ).astype(action_spec.dtype, copy=False)


def main(argv):
    del argv

    random_state = np.random.RandomState(42)

    # env = manipulation.load('stack_2_of_3_bricks_random_order_vision', seed=42)
    env = manipulation.load('stack_2_bricks_moveable_base_features', seed=42)

    print(type(env))
    

    action_spec = env.action_spec()
    # observation_spec = env.observation_spec()

    time_step = env.reset()

    # Define a uniform random policy.
    def random_policy(time_step):
        if time_step.first():  # Unused.
            print(type(time_step.observation))
            for key, value in time_step.observation.items():
                print(key, value)

        # next_action = input("Next action: ")
        # print(next_action)

        getch = _Getch()

        print(getch.__call__())

        action = np.random.uniform(low=action_spec.minimum,
                                    high=action_spec.maximum,
                                    size=action_spec.shape)

        return action

    viewer.launch(env,policy=random_policy)

    # pos=np.r_[0., 0., 0.3]
    # quat=np.r_[0., 1., 0., 1.]
    
    # arm = kinova.JacoArm()
    # hand = kinova.JacoHand()
    # arena = composer.Arena()
    # arm.attach(hand)
    # arena.attach(arm)
    # physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)


    # width = 640
    # height = 480
    # pixels = physics.render(height, width)
    # im = PIL.Image.fromarray(pixels)

    # im.show()

    # # Normalize the quaternion.
    # quat /= np.linalg.norm(quat)

    # # Drive the arm so that the pinch site is at the desired position and
    # # orientation.
    # success = arm.set_site_to_xpos(
    #     physics=physics,
    #     random_state=np.random.RandomState(0),
    #     site=hand.pinch_site,
    #     target_pos=pos,
    #     target_quat=quat)

    # pixels2 = physics.render()
    # im2 = PIL.Image.fromarray(pixels2)

    # im2.show()
    

    # # Check that the observations are as expected.
    # observed_pos = hand.observables.pinch_site_pos(physics)

    # print(observed_pos)


    # print(f"Action space min: {action_spec.minimum}, Action space max: {action_spec.maximum}")

#   timestep = env.reset()

#   for i in range(1,100):
#     timestep = env.step(action_spec.minimum)



if __name__ == '__main__':
  app.run(main)
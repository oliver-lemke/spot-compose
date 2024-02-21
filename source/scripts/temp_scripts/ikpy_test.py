import os.path
import sys

import numpy as np

import ikpy.utils.plot as plot_utils
from ikpy.chain import Chain
from matplotlib import pyplot as plt
from utils import recursive_config


def main() -> None:
    config = recursive_config.Config()
    spot_description_path = config.get_subpath("spot_description")
    arm_urdf = os.path.join(str(spot_description_path), "urdf", "spot_arm.urdf")
    arm_chain = Chain.from_urdf_file(
        arm_urdf,
        base_elements=["body"],
        active_links_mask=[False, True, True, False, True, True, True, True, True],
    )
    target_position = [-0.2, -0.5, 0]
    # target_orientation = [0, 1, 0]

    ik = arm_chain.inverse_kinematics(target_position)  # , target_orientation)
    print(arm_chain)
    print(ik)
    tolerance = 1e-5
    found_ik = not np.all(np.abs(ik) <= tolerance)
    if not found_ik:
        print("Could not find an inverse kinematics approach!")
        sys.exit(1)

    _, ax = plot_utils.init_3d_figure()
    arm_chain.plot(ik, ax, target=target_position)
    plt.show()
    sys.exit(0)


if __name__ == "__main__":
    main()

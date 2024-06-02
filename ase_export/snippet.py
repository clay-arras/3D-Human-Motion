# INSTRUCTIONS: add this snippet in ASE/ase/env/tasks/humanoid.py at the end of the function _compute_humanoid_obs() on line 417


# converting observations to euler rotations
# assumes there's only one env
from scipy.spatial.transform import Rotation
import numpy as np

ENV_ID = 0
export = []
for i in body_pos[ENV_ID][0]:
    export.append(i.item() * 10)

for joint in range(body_rot.shape[1]):
    body_rot_joint = torch.Tensor.cpu(body_rot[ENV_ID][joint])
    r = Rotation.from_quat([body_rot_joint[0], body_rot_joint[1], body_rot_joint[2], body_rot_joint[3]])
    for i in r.as_euler('xyz', degrees=True):
        export.append(i)


# saving values to file
from datetime import datetime
import os
now = datetime.now().strftime("%d_%m_%Y_%H_%M")

export_filename = f"export{now}.bvh"
export_filepath = "/home/dongsheng/REX-human-motion/ASE/ase/export/"

if not os.path.isfile(export_filepath + export_filename):
    os.system(f"cp {export_filepath + 'template.bvh'} {export_filepath + export_filename}")
file = open(export_filepath + export_filename, "a+")
file.write(' '.join([str(i) for i in export]))
file.write('\n')

import tidy3d as td
from tidy3d import web
from tidy3d.config import Env

Env.prod.active()
web.configure("CICIf8UmEdMtBSJbxW66npxujQ3Ob7Wiy4UHChijaVTAdrnu")

#Env.dev.active()
#web.configure("Ltrvqel7oCenUTH88Pqh99vn7ikCD25KFPZ0phz2Mxtgl5I4")

#Env.uat.active()
#web.configure("LmpSvRP0MGOuKOgm9ZJn97l9RE8t5I2ENTI9RLwbXlmmW89Z")


modeler = td.Tidy3dBaseModel.from_file("modal_cm.json")
task_id = web.upload(modeler, task_name="directional coupler")
from tidy3d.web.core.http_util import http
import json
# resp = http.get(
#     f"rf/task/{task_id}/statistics",
# )
#print(json.dumps(resp, indent=4))
web.run(modeler, folder_name="modal_cm")
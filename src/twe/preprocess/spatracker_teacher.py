import numpy as np
import torch

from twe.preprocess.teacher import TeacherTracks

SPATRACKER_COMMIT = "7e12274c52077860cebfe007a6290777db43b63c"
FRONT_MODEL = "Yuxihenry/SpatialTrackerV2_Front"
OFFLINE_MODEL = "Yuxihenry/SpatialTrackerV2-Offline"


def teacher_revision(width: int = 518) -> str:
    return f"spatrackerv2@{SPATRACKER_COMMIT[:12]}:offline:w{width}"


def reprojection_error(points_camera: np.ndarray, intrinsics: np.ndarray, xy: np.ndarray) -> np.ndarray:
    z = points_camera[:, 2]
    u = intrinsics[..., 0, 0] * points_camera[:, 0] / z + intrinsics[..., 0, 2]
    v = intrinsics[..., 1, 1] * points_camera[:, 1] / z + intrinsics[..., 1, 2]
    return np.where(z > 0, np.hypot(u - xy[:, 0], v - xy[:, 1]), np.inf)


class SpaTrackerTeacher:
    def __init__(self, front: torch.nn.Module, predictor: torch.nn.Module, preprocess, width: int = 518,
                 depth_confidence: float = 0.5, iters_track: int = 4, max_reprojection_px: float = 4.0,
                 device: str = "cuda"):
        self.front = front
        self.predictor = predictor
        self.preprocess = preprocess
        self.width = width
        self.depth_confidence = depth_confidence
        self.iters_track = iters_track
        self.max_reprojection_px = max_reprojection_px
        self.device = device
        self.revision = teacher_revision(width)

    @torch.no_grad()
    def track(self, frames: np.ndarray, query_xy: np.ndarray, query_frame: np.ndarray) -> TeacherTracks:
        height, width = frames.shape[1:3]
        video = self.preprocess(torch.from_numpy(frames).permute(0, 3, 1, 2).float())
        scale = np.array([video.shape[-1] / width, video.shape[-2] / height])
        if video.shape[-1] != self.width or video.shape[-2] != round(height * scale[0] / 14) * 14:
            raise ValueError(f"unexpected teacher resolution {tuple(video.shape[-2:])}")
        xy = query_xy * scale - 0.5
        with torch.autocast("cuda", dtype=torch.bfloat16):
            front = self.front(video[None].to(self.device) / 255)
        extrinsic = front["poses_pred"].squeeze(0).float().cpu().numpy()
        intrinsic = front["intrs"].squeeze(0).float().cpu().numpy()
        depth = front["points_map"][..., 2].squeeze(0).float().cpu().numpy()
        unc = front["unc_metric"].squeeze(0).float().cpu().numpy() > self.depth_confidence
        queries = np.concatenate([query_frame[:, None], xy], 1).astype(np.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c2w, intrs, point_map, conf_depth, track3d, _, vis, conf, _ = self.predictor.forward(
                video, depth=depth, intrs=intrinsic, extrs=extrinsic, queries=queries, fps=1, full_point=True,
                iters_track=self.iters_track, query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc,
                support_frame=len(video) - 1, replace_ratio=0.2)
        c2w = c2w.double().cpu().numpy()
        intrs = intrs.double().cpu().numpy()
        camera = track3d[..., :3].double().cpu().numpy()
        points_world = np.einsum("tij,tnj->nti", c2w[:, :3, :3], camera) + c2w[:, None, :3, 3].transpose(1, 0, 2)
        reliability = (vis[..., 0] * conf[..., 0]).clamp(0, 1).float().cpu().numpy().T
        frame = query_frame.astype(int)
        error = reprojection_error(camera[frame, np.arange(len(xy))], intrs[frame], xy)
        if np.median(error) > self.max_reprojection_px:
            raise ValueError(f"teacher reprojection error {np.median(error):.2f}px breaks opencv convention check")
        reliability[error > self.max_reprojection_px, :] = 0.0
        depth_maps = point_map[:, 2].float().cpu().numpy()
        depth_maps[conf_depth.float().cpu().numpy() < self.depth_confidence] = np.nan
        lift = np.diag([1 / scale[0], 1 / scale[1], 1.0])
        source_k = intrs.copy()
        source_k[:, :2, 2] += 0.5
        source_k = lift @ source_k
        return TeacherTracks(points_world, reliability, np.linalg.inv(c2w), depth_maps, "opencv", self.revision,
                             source_k)


def load_spatracker(device: str = "cuda") -> SpaTrackerTeacher:
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

    front = VGGT4Track.from_pretrained(FRONT_MODEL).eval().to(device)
    predictor = Predictor.from_pretrained(OFFLINE_MODEL)
    predictor.eval()
    predictor.to(device)
    return SpaTrackerTeacher(front, predictor, preprocess_image, device=device)

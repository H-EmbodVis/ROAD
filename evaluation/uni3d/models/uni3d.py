# Modified from Uni3D: reduced to point-encoder construction for Uni3D-I.
import timm

from .point_encoder import PointcloudEncoder


def create_uni3d(args):
    point_transformer = timm.create_model(
        args.pc_model,
        checkpoint_path=args.pretrained_pc,
        drop_path_rate=args.drop_path_rate,
    )
    return PointcloudEncoder(point_transformer, args)

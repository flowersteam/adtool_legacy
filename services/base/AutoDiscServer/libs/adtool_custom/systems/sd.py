import torch
from adtool_custom.maps.IdentityBehaviorMap import IdentityBehaviorMap
from adtool_custom.maps.TextToVectorMap import TextToVectorMap
from adtool_custom.systems.StableDiffusionPropagator import StableDiffusionPropagator


def test():
    txt_embedder = TextToVectorMap(
        seed_prompt="a watercolor painting of Bordeaux, France"
    )
    sd = StableDiffusionPropagator(num_inference_steps=16)
    id = IdentityBehaviorMap()

    # get initial seed image
    data = txt_embedder.map({}, use_seed_vector=True)
    print(f"size: {data['params'].size()}")
    data = sd.map(data)
    data = id.map(data)
    img = sd._decode_image(data["latent_vector"])
    img.save(f"out_seed.png")

    # generate random samples
    for i in range(10):
        # this first step will sample around the seed_prompt
        data = txt_embedder.map(data)

        # reverse diffuse here
        data = sd.map(data, fix_seed=False)
        # save the video
        byte_vid = sd.render({})
        with open(f"outvid_{i}.png", "wb") as f:
            f.write(byte_vid)

        # does nothing, because there is no IMGEP
        data = id.map(data)

        img = sd._decode_image(data["latent_vector"])
        img.save(f"out_{i}.png")


if __name__ == "__main__":
    test()
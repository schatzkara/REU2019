from torchsummary import summary
from phase0.modifiedVGG import vgg16
from phase0.modifiedI3D import InceptionI3d
from phase0.generator import Generator

image_size = (3, 112, 112)
video_size = (3, 16, 112, 112)

if __name__ == "__main__":
    print_summary = True

    vgg = vgg16()
    i3d = InceptionI3d(final_endpoint='Mixed_5c', in_frames=video_size[1])
    gen = Generator(in_channels=1536, out_frames=video_size[1])
    # model = FullNetwork(output_shape=(1,) + video_size)

    if print_summary:
        summary(vgg, input_size=image_size)
        summary(i3d, input_size=video_size)
        summary(gen, input_size=(1536, 7, 7))
        # summary(model, input_size=[video_size, video_size, image_size, image_size])

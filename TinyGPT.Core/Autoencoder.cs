using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TinyGPT.Core;
using TorchSharp;
using TorchSharp.Modules;


namespace TinyGPT{
	public sealed class EncoderStep : Module<Tensor, Tensor>
	{
		private readonly Module<Tensor, Tensor> conv;
		private readonly Module<Tensor, Tensor> activation;
		public EncoderStep(long inputChannel, long outputChannel, long kernelSize, string name) : base(name)
		{
			conv = Conv1d(inputChannel, outputChannel, kernelSize, 2);
			activation = new LeakySoftplus("");
			RegisterComponents();

		}

		public override Tensor forward(Tensor input)
		{
			return activation.forward(conv.forward(input));
		}
	}
	public sealed class DecoderStep : Module<Tensor, Tensor>
	{
		private readonly Module<Tensor, Tensor> deconv;
		private readonly Module<Tensor, Tensor> activation;

		public DecoderStep(long inputChannel, long outputChannel, long kernelSize, string name) : base(name)
		{
			deconv = ConvTranspose1d(inputChannel, outputChannel, kernelSize, 2);
			activation = new LeakySoftplus("");
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			return activation.forward(deconv.forward(input));
		}
	}
	public sealed class Encoderv1 : Module<Tensor, (Tensor, Tensor)>
	{

		private readonly Module<Tensor, Tensor> layer1 = new EncoderStep(1, 2, 4, "");
		private readonly Module<Tensor, Tensor> layer2 = new EncoderStep(2, 4, 4, "");
		private readonly Module<Tensor, Tensor> layer3 = new EncoderStep(4, 8, 4, "");
		private readonly Module<Tensor, Tensor> layer4 = new EncoderStep(8, 16, 4, "");
		private readonly Module<Tensor, Tensor> layer5 = new EncoderStep(16, 32, 4, "");
		private readonly Module<Tensor, Tensor> layer6 = new EncoderStep(32, 64, 4, "");
		private readonly Module<Tensor, Tensor> layer7 = new EncoderStep(64, 128, 4, "");
		private readonly Module<Tensor, Tensor> layer8 = new EncoderStep(128, 256, 4, "");
		private readonly Module<Tensor, Tensor> layer9 = new EncoderStep(256, 512, 2, "");
		private readonly Module<Tensor, Tensor> layer10 = new EncoderStep(512, 1024, 1, "");
		private readonly Module<Tensor, Tensor> layer11 = new DenseStep(1024, 1024, "");
		private readonly Module<Tensor, Tensor> layer12 = new DenseStep(1024, 1024, "");
		private readonly Module<Tensor, Tensor> meanlayer = Linear(1024,512);
		private readonly Module<Tensor, Tensor> stddevlayer = Linear(1024, 512);
		private readonly Module<Tensor, Tensor> stddevactivation = Softplus();
		public Encoderv1(string name) : base(name)
		{
			RegisterComponents();

		}
		private static readonly long[] review = new long[] { 1, -1 };
		private static readonly long[] review2 = new long[] { -1 };

		public override (Tensor, Tensor) forward(Tensor input)
		{
			input = input.view(review);
			Tensor x = layer12.forward(layer11.forward(layer10.forward(layer9.forward(layer8.forward(layer7.forward(layer6.forward(layer5.forward(layer4.forward(layer3.forward(layer2.forward(layer1.forward(input)))))))))).view(review2)));
			return (meanlayer.forward(x), stddevactivation.forward(stddevlayer.forward(x)) + 0.00390625f);
		}
	}
	public sealed class Decoderv1 : Module<Tensor, Tensor>
	{
		private static readonly long[] review = new long[] { 1024, -1 };
		private static readonly long[] review2 = new long[] { -1 };

		private readonly Module<Tensor, Tensor> layer1 = new DenseStep(512, 1024, "");
		private readonly Module<Tensor, Tensor> layer2 = new DenseStep(1024, 1024, "");
		private readonly Module<Tensor, Tensor> layer3 = new DenseStep(1024, 1024, "");
		private readonly Module<Tensor, Tensor> layer4 = new DecoderStep(1024, 512, 2, "");
		private readonly Module<Tensor, Tensor> layer5 = new DecoderStep(512, 256, 2, "");
		private readonly Module<Tensor, Tensor> layer6 = new DecoderStep(256, 128, 2, "");
		private readonly Module<Tensor, Tensor> layer7 = new DecoderStep(128, 64, 2, "");
		private readonly Module<Tensor, Tensor> layer8 = new DecoderStep(64, 32, 2, "");
		private readonly Module<Tensor, Tensor> layer9 = new DecoderStep(32, 16, 2, "");
		private readonly Module<Tensor, Tensor> layer10 = new DecoderStep(16, 8, 2, "");
		private readonly Module<Tensor, Tensor> layer11 = new DecoderStep(8, 4, 2, "");
		private readonly Module<Tensor, Tensor> layer12 = new DecoderStep(4, 2, 2, "");
		private readonly Module<Tensor, Tensor> layer13 = new DecoderStep(2, 1, 2, "");

		public Decoderv1(string name) : base(name)
		{
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			return layer13.forward(layer12.forward(layer11.forward(layer10.forward(layer9.forward(layer8.forward(layer7.forward(layer6.forward(layer5.forward(layer4.forward(layer3.forward(layer2.forward(layer1.forward(input))).view(review))))))))))).view(review2);
		}
	}

	public sealed class VAECombinedNetwork : Module<Tensor, (Tensor, Tensor, Tensor)>
	{
		private readonly Encoderv1 encoderv1;
		private readonly Decoderv1 decoderv1;

		public VAECombinedNetwork(Encoderv1 encoderv1, Decoderv1 decoderv1, string name) : base(name)
		{
			this.encoderv1 = encoderv1 ?? throw new ArgumentNullException(nameof(encoderv1));
			this.decoderv1 = decoderv1 ?? throw new ArgumentNullException(nameof(decoderv1));
			RegisterComponents();
		}

		public override (Tensor, Tensor, Tensor) forward(Tensor input)
		{
			(Tensor a, Tensor b) = encoderv1.forward(input);
			return (decoderv1.forward(normal(a, b)), a, b);
		}
	}
}
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Parameter = TorchSharp.Modules.Parameter;

namespace TinyGPT.Core
{
	public abstract class FullGPTDecoderUnit : Module<ReadOnlyMemory<ushort>, Tensor>
	{
		protected FullGPTDecoderUnit(string name) : base(name)
		{
		}

		protected FullGPTDecoderUnit(nint handle, nint boxedHandle) : base(handle, boxedHandle)
		{
		}

		public abstract Tensor Forward(ReadOnlySpan<ushort> input);
		public sealed override Tensor forward(ReadOnlyMemory<ushort> input)
		{
			return Forward(input.Span);
		}
	}


	public sealed class GPTDecoderUnitV1 : FullGPTDecoderUnit
	{

		private sealed class GPTHiddenLayer : Module<Tensor, Tensor>
		{
			private readonly Parameter residual;
			private readonly LayerNorm layerNorm;
			private readonly Linear tail;
			private readonly Module<Tensor, Tensor> module;
			public GPTHiddenLayer(Module<Tensor, Tensor> module, int size, double epsilon) : base("")
			{
				residual = Parameter(ones(1, size));
				layerNorm = LayerNorm(size, epsilon, false);
				tail = Misc.CreateXavierInitializedLinear(size, size, true);
				this.module = module;
				RegisterComponents();
			}

			public sealed override Tensor forward(Tensor input)
			{
				using (NewDisposeScope())
				{
					Tensor x;
					using (Tensor y = module.forward(input))
					{
						x = tail.forward(y);
					}
					Tensor z;
					using (Tensor y = input.mul(residual))
					{
						using (x)
						{
							z = y.add(x);
						}
					}
					using (z)
					{
						return layerNorm.forward(z).MoveToOuterDisposeScope();
					}

				}
			}
		}
		private sealed class GPTAttentionLayer : Module<Tensor, Tensor>
		{
			private readonly int headcount;
			private readonly int latentTokenSize;
			private readonly ModuleList<AttentionLayer> attentionHeads = new ModuleList<AttentionLayer>();
			public GPTAttentionLayer(int latentTokenSize, int heads) : base("")
			{
				headcount = heads;
				this.latentTokenSize = latentTokenSize;
				for (int i = 0; i < heads; ++i)
				{
					attentionHeads.Add(new AttentionLayer("", latentTokenSize));
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				int heads = headcount;
				int latentTokenSize = this.latentTokenSize;
				Tensor[] attnheads = input.split(latentTokenSize, 1);
				for (int i = 0; i < heads; ++i)
				{
					using Tensor x = attnheads[i];
					attnheads[i] = attentionHeads[i].forward(x);
				}
				try
				{
					return cat(attnheads, 1);
				}
				finally
				{
					for (int i = 0; i < heads; ++i)
					{
						attnheads[i].Dispose();
					}
				}

			}
		}
		private sealed class GPTComputeLayer : Module<Tensor, Tensor>
		{
			private readonly Linear linear;
			public GPTComputeLayer(int size) : base("")
			{
				linear = Misc.CreateXavierInitializedLinear(size, size, true);
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				using Tensor x = linear.forward(input);
				return x.gelu().MoveToOuterDisposeScope();
			}
		}




		private readonly Linear outputEmbedding;
		private readonly ModuleList<GPTHiddenLayer> layers = new ModuleList<GPTHiddenLayer>();
		private readonly Tensor wordEmbedding;


		private readonly double scale;



		private readonly Parameter positionalEncodingBias;
		private readonly Parameter positionalEncodingWeight;
		private readonly int headcount;
		//private readonly LayerNorm layerNorm;
		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int tokenClasses, int coreDepth, double initialFrequency, double epsilon) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}
			int fullsize = latentTokenSize * attentionHeadsCount;
			positionalEncodingWeight = Parameter(randn(fullsize, 1).mul_(initialFrequency));
			positionalEncodingBias = Parameter(zeros(fullsize, 1));

			for (int i = 0; i < coreDepth; ++i)
			{
				layers.Add(new GPTHiddenLayer(new GPTAttentionLayer(latentTokenSize, attentionHeadsCount), fullsize, epsilon));
				layers.Add(new GPTHiddenLayer(new GPTComputeLayer(fullsize), fullsize, epsilon));
			}
			layers.Add(new GPTHiddenLayer(new GPTAttentionLayer(latentTokenSize, attentionHeadsCount), fullsize, epsilon));

			outputEmbedding = Misc.CreateXavierInitializedLinear(fullsize, latentTokenSize, true);

			wordEmbedding = randn(latentTokenSize, tokenClasses);
			scale = Math.Sqrt(latentTokenSize * 2);
			headcount = attentionHeadsCount;
			//layerNorm = LayerNorm(tokenClasses, epsilon);
			RegisterComponents();
		}

		public Tensor Forward(ReadOnlySpan<ushort> input, int slice)
		{
			int len = input.Length;
			if (len == 0)
			{
				throw new IndexOutOfRangeException(nameof(input));
			}
			using (NewDisposeScope())
			{

				int headcount = this.headcount;
				Tensor[] all = new Tensor[len];
				Tensor[] heads2 = new Tensor[headcount];
				for (int i = 0; i < len; ++i)
				{
					long my = input[i];
					Tensor n;
					using (Tensor p = positionalEncodingWeight.mul(i))
					{
						n = p.add(positionalEncodingBias);
					}
					using (Tensor p = n)
					{
						n = p.sin();
					}

					Tensor z2;
					using (Tensor z = wordEmbedding.slice(1, my, my + 1, 1))
					{
						for (int p = 0; p < headcount; ++p)
						{
							heads2[p] = z;
						}
						z2 = cat(heads2, 0);
					}
					using (z2)
					{
						using (n)
						{
							all[i] = z2.add(n);
						}
					}

				}

				Tensor y;
				using (Tensor c2 = cat(all, 1))
				{
					y = c2.transpose(0, 1);
				}



				foreach (GPTHiddenLayer hiddenLayer in layers)
				{
					using Tensor x = y;
					y = hiddenLayer.forward(x);
				}

				using (Tensor x = y)
				{
					y = x.slice(0, slice, len, 1);
				}
				using (Tensor x = y)
				{

					y = outputEmbedding.forward(x);
				}

				using (Tensor x = y)
				{
					y = x.matmul(wordEmbedding);
				}
				using (y)
				{
					return y.div(scale).MoveToOuterDisposeScope();
				}
			}
		}

		public override Tensor Forward(ReadOnlySpan<ushort> input)
		{
			using (NewDisposeScope())
			{
				using (Tensor x = Forward(input, input.Length - 1))
				{
					return x.squeeze(0).MoveToOuterDisposeScope();
				}
			}
		}
	}


}

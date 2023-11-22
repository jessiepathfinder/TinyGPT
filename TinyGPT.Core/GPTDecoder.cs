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


	public sealed class GPTDecoderUnitV1 : FullGPTDecoderUnit, IL1Regularizable
	{






		private readonly ModuleList<Module<Tensor, Tensor>> layers = new ModuleList<Module<Tensor, Tensor>>();
		private readonly Parameter wordEmbedding;
		private readonly Parameter finalscale;



		private readonly Parameter positionalEncodingBias;
		private readonly Parameter positionalEncodingWeight;
		private readonly int headcount;
		//private readonly LayerNorm layerNorm;
		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int tokenClasses, int coreDepth, double initialFrequency, int attentionKeySize, int attentionValueSize, int computecoresize, double epsilon) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}
			positionalEncodingWeight = Parameter(randn(latentTokenSize).mul_(initialFrequency));
			positionalEncodingBias = Parameter(zeros(latentTokenSize));
			finalscale = Parameter(full(1, latentTokenSize, (Scalar)(1.0 / Math.Sqrt(latentTokenSize))));

			for (int i = 0; i < coreDepth; ++i)
			{
				layers.Add(new MultiheadResidualAttention("", latentTokenSize, attentionKeySize, attentionValueSize, latentTokenSize, attentionHeadsCount, epsilon));
				layers.Add(new ResidualComputeLayer("", latentTokenSize, computecoresize, epsilon));
			}


			wordEmbedding = Parameter(randn(latentTokenSize, tokenClasses));
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
				Tensor wordEmbedding = this.wordEmbedding;


				Tensor y;
				using (NewDisposeScope())
				{
					for (int i = 0; i < len; ++i)
					{
						Tensor y2;
						using(Tensor x2 = positionalEncodingWeight.mul(i)){
							y2 = x2.add(positionalEncodingBias);
						}
						using(Tensor x2 = y2){
							y2 = x2.sin();
						}
						Tensor slice2;
						using (Tensor slice3 = wordEmbedding.select(1, input[i]))
						{
							using(y2){
								slice2 = slice3.add(y2);
							}
						}
						
						using (slice2)
						{
							all[i] = slice2.unsqueeze(0);
						}

					}
					y = cat(all, 0).MoveToOuterDisposeScope();
				}



				using (Tensor mask = Transformer.CreateCausalAttentionMask(len, len, wordEmbedding.dtype, wordEmbedding.device))
				{
					foreach (Module<Tensor, Tensor> hiddenLayer in layers)
					{
						using Tensor x = y;
						if (hiddenLayer is MultiheadResidualAttention multiheadResidualAttention)
						{
							y = multiheadResidualAttention.Forward(x, x, mask);
						}
						else
						{
							y = hiddenLayer.forward(x);
						}
					}
				}


				using (Tensor x = y)
				{
					y = x.slice(0, slice, len, 1);
				}

				using (Tensor x = y)
				{
					y = x.mul(finalscale);
				}
				using (y)
				{
					return y.matmul(wordEmbedding).MoveToOuterDisposeScope();
				}
			}
		}

		public override Tensor Forward(ReadOnlySpan<ushort> input)
		{
			using (NewDisposeScope())
			{
				using Tensor x = Forward(input, input.Length - 1);
				return x.squeeze(0).MoveToOuterDisposeScope();
			}
		}

		public void L1Regularize(double lambda)
		{
			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL1Regularizable regularizable)
				{
					regularizable.L1Regularize(lambda);
				}
			}
		}
	}


}
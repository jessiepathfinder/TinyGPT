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


	public sealed class GPTDecoderUnitV1 : FullGPTDecoderUnit, IL2Regularizable
	{






		private readonly ModuleList<Module<Tensor, Tensor>> layers = new ModuleList<Module<Tensor, Tensor>>();
		private readonly Parameter wordEmbedding;
		public readonly Linear defaultEngine;
		private readonly ResidualAutogatedMultiQueryAttention finalattention;
		private readonly ResidualAutogatedComputeLayer finalCompute;


		private readonly Parameter positionalEncodingBias;
		private readonly Parameter positionalEncodingWeight;
		private readonly int headcount;
		private readonly int widthMultiplier;
		//private readonly Parameter finalBias;
		private readonly Scalar scale;
		//private readonly LayerNorm layerNorm;
		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int tokenClasses, int coreDepth, double initialFrequency, int attentionKeySize, int attentionValueSize, int computecoresize, int widthMultiplier1, double epsilon) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}
			widthMultiplier = widthMultiplier1;
			int multipliedWidth = latentTokenSize * widthMultiplier1;
			positionalEncodingWeight = Parameter(randn(multipliedWidth).mul_(initialFrequency));
			positionalEncodingBias = Parameter(zeros(multipliedWidth));
			Span<long> longs = stackalloc long[2];
			longs[0] = multipliedWidth;
			longs[1] = latentTokenSize;
			defaultEngine = Misc.CreateXavierInitializedLinear(multipliedWidth, latentTokenSize, false);
			finalCompute = new ResidualAutogatedComputeLayer("", multipliedWidth, computecoresize, epsilon);

			scale = 1.0 / Math.Sqrt(tokenClasses);

			for (int i = 0; i < coreDepth; ++i)
			{
				layers.Add(new ResidualAutogatedMultiQueryAttention("", multipliedWidth, attentionKeySize, attentionValueSize, attentionHeadsCount, epsilon));
				layers.Add(new ResidualAutogatedComputeLayer("", multipliedWidth, computecoresize, epsilon));
			}
			finalattention = new ResidualAutogatedMultiQueryAttention("", multipliedWidth, attentionKeySize, attentionValueSize, attentionHeadsCount, epsilon);

			//finalBias = Parameter(zeros(1, tokenClasses));
			wordEmbedding = Parameter(randn(latentTokenSize, tokenClasses));
			headcount = attentionHeadsCount;

			//layerNorm = LayerNorm(tokenClasses, epsilon);
			RegisterComponents();
		}
		public Tensor Encode(ReadOnlySpan<ushort> input, int slice)
		{
			int len = input.Length;
			if (len == 0)
			{
				throw new IndexOutOfRangeException(nameof(input));
			}
			using (NewDisposeScope())
			{

				int headcount = this.headcount;
				int widthMultiplier = this.widthMultiplier;
				Tensor[] combine = new Tensor[widthMultiplier];


				Tensor[] all = new Tensor[len];
				Tensor wordEmbedding = this.wordEmbedding;


				Tensor y;
				using (NewDisposeScope())
				{
					for (int i = 0; i < len; ++i)
					{
						Tensor y2;
						using (Tensor x2 = positionalEncodingWeight.mul(i))
						{
							y2 = x2.add(positionalEncodingBias);
						}
						using (Tensor x2 = y2)
						{
							y2 = x2.sin();
						}
						Tensor slice2;
						Tensor y4;
						using (NewDisposeScope())
						{
							using Tensor slice3 = wordEmbedding.select(1, input[i]);
							for (int k = 0; k < widthMultiplier; ++k)
							{
								combine[k] = slice3;
							}
							y4 = cat(combine, 0).MoveToOuterDisposeScope();
						}
						using (y2)
						{
							using (y4)
							{
								slice2 = y4.add(y2);
							}
						}

						using (slice2)
						{
							all[i] = slice2.unsqueeze(0);
						}

					}
					y = cat(all, 0).MoveToOuterDisposeScope();
				}



				using (Tensor mask = Transformer.CreateCausalAttentionMask(len, len, ScalarType.Float64, wordEmbedding.device))
				{
					foreach (Module<Tensor, Tensor> hiddenLayer in layers)
					{
						using Tensor x = y;
						if (hiddenLayer is ResidualAutogatedMultiQueryAttention multiheadResidualAttention)
						{
							y = multiheadResidualAttention.Forward(x, x, mask);
						}
						else
						{
							y = hiddenLayer.forward(x);
						}
					}
					if (slice == 0)
					{
						using Tensor x = y;
						y = finalattention.Forward(x, x, mask);
					}
				}


				if (slice > 0)
				{
					using Tensor x = y.slice(0, slice, len, 1), x2 = y, mask = Transformer.CreateCausalAttentionMask(len - slice, len, ScalarType.Float64, wordEmbedding.device);
					y = finalattention.Forward(x, x2, mask);
				}
				using (y)
				{
					return finalCompute.forward(y).MoveToOuterDisposeScope();
				}
			}
		}
		public Tensor Decode(Tensor y){

			using (NewDisposeScope()){
				using (Tensor x = y)
				{
					y = x.mul(scale);
				}
				using (y)
				{
					return y.matmul(wordEmbedding).MoveToOuterDisposeScope();
				}

			}
		}

		public Tensor Forward(ReadOnlySpan<ushort> input, int slice)
		{
			using(NewDisposeScope()){
				Tensor y;
				using (Tensor x = Encode(input, slice))
				{
					y = defaultEngine.forward(x);
				}
				using(y){
					return Decode(y).MoveToOuterDisposeScope();
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


		public void L2Regularize(double lambda)
		{
			//Misc.L2RegularizeIMPL(finalcompress.weight, lambda);
			finalCompute.L2Regularize(lambda);
			foreach (Module<Tensor, Tensor> layer in layers)
			{
				if (layer is IL2Regularizable regularizable)
				{
					regularizable.L2Regularize(lambda);
				}
			}
		}
		
	}


}
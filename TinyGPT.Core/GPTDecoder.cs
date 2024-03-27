using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using Tensorboard;
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
		private readonly Linear defaultEngine;
		private readonly MultiheadSelfAttention sharedattention;
		private readonly ResidualComputeLayer finalCompute;
		private readonly int widthMultiplier;


		private readonly Parameter positionalEncodingBias;
		private readonly Parameter positionalEncodingWeight;
		private readonly Parameter finalBias;
		private readonly int headcount;
		private readonly Scalar scale;
		private readonly Scalar epsilon;
		private readonly Linear expand;
		private static readonly Scalar zero = 0;


		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int tokenClasses, int coreDepth, double positionalEncodingConstant, int attentionValueSize, int widthMultiplier1, double epsilon, long geluCoreUnits,long rnnDepth, int rnnMemorySize) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}
			widthMultiplier = widthMultiplier1;
			int multipliedWidth = latentTokenSize * widthMultiplier1;
			double[] positionalEncodingWeights = new double[multipliedWidth];
			double[] positionalEncodingBiases = new double[multipliedWidth];
			double dml = multipliedWidth;
			for(int i = 0; i < multipliedWidth; ++i){
				double d = 1.0 / Math.Pow(positionalEncodingConstant, (i/2) / dml);
				positionalEncodingWeights[i] = d;
				positionalEncodingBiases[i] = ((i % 2) * d * double.Pi) / 2.0;
			}

			positionalEncodingWeight = Parameter(tensor(positionalEncodingWeights));
			positionalEncodingBias = Parameter(tensor(positionalEncodingBiases));
			expand = Misc.CreateKaimingInitializedLinear(latentTokenSize, multipliedWidth, true, init.FanInOut.FanIn);

			defaultEngine = Misc.CreateKaimingInitializedLinear(multipliedWidth, latentTokenSize, false, init.FanInOut.FanIn);
			finalCompute = new ResidualComputeLayer("", multipliedWidth, epsilon, geluCoreUnits);

			scale = 1.0 / Math.Sqrt(tokenClasses);
			for (int i = 0; i < rnnDepth; ++i)
			{
				//layers.Add(new ResidualCausalConvolationalLookback("", multipliedWidth, latentTokenSize, widthMultiplier, epsilon));
				//layers.Add(new MultiheadSelfAttention("", multipliedWidth, attentionValueSize, attentionHeadsCount, epsilon, true));
				layers.Add(new TinyRNN("", multipliedWidth, rnnMemorySize, epsilon));
			}
			for (int i = 0; i < coreDepth; ++i)
			{
				//layers.Add(new ResidualCausalConvolationalLookback("", multipliedWidth, latentTokenSize, widthMultiplier, epsilon));
				//layers.Add(new MultiheadSelfAttention("", multipliedWidth, attentionValueSize, attentionHeadsCount, epsilon, true));
				layers.Add(new ResidualComputeLayer("", multipliedWidth, epsilon, geluCoreUnits));
			}
			//layers.Add(new ResidualCausalConvolationalLookback("", multipliedWidth, latentTokenSize, widthMultiplier, epsilon));
			//layers.Add(new TinyRNN("", multipliedWidth, multipliedWidth, epsilon));
			sharedattention = new MultiheadSelfAttention("", multipliedWidth, attentionValueSize, attentionHeadsCount, epsilon, false);
			finalBias = Parameter(zeros(tokenClasses));

			//finalBias = Parameter(zeros(1, tokenClasses));
			wordEmbedding = Parameter(randn(latentTokenSize, tokenClasses));
			headcount = attentionHeadsCount;

			//layerNorm = LayerNorm(tokenClasses, epsilon);
			this.epsilon = epsilon;
			RegisterComponents();
		}
		public Tensor Encode(ReadOnlySpan<ushort> input, int slice, double dropout)
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


				Tensor[] all = new Tensor[len];
				Tensor wordEmbedding = this.wordEmbedding;


				Tensor y;
				using (NewDisposeScope())
				{
					for (int i = 0; i < len; ++i)
					{
						long z = input[i];
						all[i] = wordEmbedding.slice(1, z, z + 1, 1);
					}
					y = cat(all, 1).MoveToOuterDisposeScope();
				}
				using (Tensor x = y)
				{
					y = x.transpose(0, 1);
				}
				using (Tensor x = y)
				{
					y = expand.forward(x);
				}
				Device device = wordEmbedding.device;
				Tensor z2;
				using(Tensor x = arange(len, ScalarType.Int32, device, false)){
					z2 = x.to(wordEmbedding.dtype);
				}
				using (Tensor x = z2)
				{
					z2 = x.unsqueeze(1);
				}
				using (Tensor x = z2){
					z2 = positionalEncodingBias.addcmul(z2, positionalEncodingWeight, zero);
				}
				using(Tensor x = z2){
					z2 = x.sin();
				}
				using(z2){
					using Tensor x = y;
					y = x.add(z2);
				}


				ResidualComputeLayer finalCompute = this.finalCompute;
				MultiheadSelfAttention finalattention = this.sharedattention;


				using (Tensor x = y)
				{
					y = CustomActivations.Norm(x, epsilon);
				}

				using (Tensor mask = Transformer.CreateCausalAttentionMask(len, len, ScalarType.Float32, device))
				{
					foreach (Module<Tensor, Tensor> hiddenLayer in layers)
					{
						using (Tensor x = y){
							y = finalattention.Forward(x, 0, mask, dropout);
						}
						using (Tensor x = y){
							y = (hiddenLayer).forward(x);
						}
						
					}
					if (slice == 0)
					{
						using Tensor x = y;
						y = finalattention.Forward(x, 0, mask, dropout);
					}
				}


				if (slice > 0)
				{
					using Tensor x2 = y, mask = Transformer.CreateCausalAttentionMask(len - slice, len, ScalarType.Float32, wordEmbedding.device);
					y = finalattention.Forward(x2, slice, mask, dropout);
				}

				using (y)
				{
					return finalCompute.forward(y).MoveToOuterDisposeScope();
				}
				
			}
		}
		
		public Tensor Decode(Tensor y)
		{

			using (NewDisposeScope())
			{
				using (Tensor x = y)
				{
					y = x.mul(scale);
				}
				using (Tensor x = y)
				{
					y = x.matmul(wordEmbedding);
				}
				using (y)
				{
					return y.add(finalBias).MoveToOuterDisposeScope();
				}

			}
		}
		private static readonly Scalar one = 1.0;
		public Tensor DefaultDecode(Tensor y)
		{
			using Tensor x = defaultEngine.forward(y);
			return Decode(x);
		}

		public Tensor Forward(ReadOnlySpan<ushort> input, int slice)
		{
			using (NewDisposeScope())
			{
				using Tensor x = Encode(input, slice, 0.0);
				return DefaultDecode(x).MoveToOuterDisposeScope();

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


		public void L2Regularize(Scalar lambda)
		{
			//Misc.L2RegularizeIMPL(finalcompress.weight, lambda);
			//finalCompute.L2Regularize(lambda);
			sharedattention.L2Regularize(lambda);
			//Misc.L2RegularizeIMPL(preFinalAccumulationGate.weight, lambda);
			//Misc.L2RegularizeIMPL(preFinalAccumulationInput.weight, lambda);

			//Misc.L2RegularizeIMPL(shortRangeAttnBoost.weight, lambda);
			//Misc.L2RegularizeIMPL(shortRangeAttnCompress.weight, lambda);
		}


	}


}
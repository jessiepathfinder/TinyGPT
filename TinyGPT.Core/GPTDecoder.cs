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





		private readonly ModuleList<Module> layers = new ModuleList<Module>();
		private readonly Parameter wordEmbedding;
		private readonly Linear defaultEngine;
		private readonly LightweightMultiheadSelfAttention finalattention;
		private readonly ResidualComputeLayer finalCompute;
		private readonly int extraRecurrence;


		private readonly Parameter positionalEncodingBias;
		private readonly Parameter positionalEncodingWeight;
		private readonly Parameter finalBias;
		private readonly Parameter finalGate;
		private readonly int headcount;
		private readonly int widthMultiplier;
		private readonly Scalar scale;
		private readonly Linear gigaDecoder;
		private readonly Linear preFinalAccumulationInput;
		private readonly Linear preFinalAccumulationGate;
		private readonly Linear finalMix;
		private readonly Scalar epsilon;
		

		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int tokenClasses, int coreDepth, double initialFrequency, int attentionValueSize, int widthMultiplier1, double epsilon, int extraRecurrence, double optimizerAssistNormStrength, long arluCoreUnits) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}
			if (extraRecurrence < 1)
			{
				throw new ArgumentNullException(nameof(extraRecurrence));
			}
			widthMultiplier = widthMultiplier1;
			int multipliedWidth = latentTokenSize * widthMultiplier1;
			Span<long> longs = stackalloc long[2];
			longs[0] = multipliedWidth;
			longs[1] = latentTokenSize;
			Span<long> mwo = longs[..1];
			positionalEncodingWeight = Parameter(normal(0.0, initialFrequency, mwo));
			positionalEncodingBias = Parameter(normal(0.0, double.Pi / initialFrequency, mwo));
			gigaDecoder = Misc.CreateXavierInitializedLinear(multipliedWidth, tokenClasses, true);
			
			defaultEngine = Misc.CreateKaimingInitializedLinear(multipliedWidth, latentTokenSize, false, init.FanInOut.FanIn);
			finalCompute = new ResidualComputeLayer("", multipliedWidth, epsilon, arluCoreUnits);

			scale = 1.0 / Math.Sqrt(tokenClasses);

			for (int i = 0; i < coreDepth; ++i)
			{
				layers.Add(new ResidualCausalConvolationalLookback("", multipliedWidth, latentTokenSize, widthMultiplier, epsilon));
				layers.Add(new LightweightMultiheadSelfAttention("", multipliedWidth, attentionValueSize, attentionHeadsCount, epsilon, optimizerAssistNormStrength));
				layers.Add(new ResidualComputeLayer("", multipliedWidth, epsilon, arluCoreUnits));
			}
			layers.Add(new ResidualCausalConvolationalLookback("", multipliedWidth, latentTokenSize, widthMultiplier, epsilon));
			finalattention = new LightweightMultiheadSelfAttention("", multipliedWidth, attentionValueSize, attentionHeadsCount, epsilon, optimizerAssistNormStrength);
			finalBias = Parameter(zeros(tokenClasses));

			//finalBias = Parameter(zeros(1, tokenClasses));
			wordEmbedding = Parameter(randn(latentTokenSize, tokenClasses));
			headcount = attentionHeadsCount;

			//layerNorm = LayerNorm(tokenClasses, epsilon);
			preFinalAccumulationInput = Misc.CreateXavierInitializedLinear(multipliedWidth, multipliedWidth, false);
			preFinalAccumulationGate = Misc.CreateXavierInitializedLinear(multipliedWidth, multipliedWidth, true);
			this.epsilon = epsilon;
			bias = Parameter(zeros(multipliedWidth));
			finalGate = Parameter(ones(multipliedWidth));
			finalMix = Misc.CreateXavierInitializedLinear(multipliedWidth, multipliedWidth, false);
			RegisterComponents();
			this.extraRecurrence = extraRecurrence;
		}
		private readonly Parameter bias;
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


				ResidualComputeLayer finalCompute = this.finalCompute;
				LightweightMultiheadSelfAttention finalattention = this.finalattention;
				

				using (Tensor x = y)
				{
					y = CustomActivations.Norm(x, epsilon);
				}

				Tensor? accumulate = null;
				using (Tensor mask = Transformer.CreateCausalAttentionMask(len, len, ScalarType.Float32, wordEmbedding.device))
				{
					for(int i = 0; i < extraRecurrence; ++i){
						foreach (Module hiddenLayer in layers)
						{
							
							if (hiddenLayer is LightweightMultiheadSelfAttention multiheadResidualAttention)
							{
								using Tensor x = y;
								y = multiheadResidualAttention.Forward(x, 0, mask, dropout);
							}

							else
							{
								using Tensor x = y;
								y = ((Module<Tensor, Tensor>)hiddenLayer).forward(x);
							}
						}
						using (Tensor x = y){
							y = finalattention.Forward(x, 0, mask);
						}
						using(Tensor x = y){
							y = finalCompute.forward(x);
						}
						Tensor sliced;
						if(slice > 0){
							using Tensor x = y.slice(0, slice, len, 1);
							sliced = preFinalAccumulationInput.forward(x);
						} else{
							sliced = preFinalAccumulationInput.forward(y);
						}


						using(sliced){
							FinalAccumulate(ref accumulate, sliced);
						}
					}
					foreach (Module hiddenLayer in layers)
					{
						if (hiddenLayer is LightweightMultiheadSelfAttention multiheadResidualAttention)
						{
							using Tensor x = y;
							y = multiheadResidualAttention.Forward(x, 0, mask, dropout);
						}
						else
						{
							using Tensor x = y;
							y = ((Module<Tensor, Tensor>)hiddenLayer).forward(x);
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

				using(Tensor x = y){
					y = finalCompute.forward(x);
				}
				FinalAccumulate(ref accumulate, y);
				using (Tensor x = accumulate ?? throw new Exception("South vietnamese lesbian not asmr yuri enough (should not reach here)")){
					accumulate = finalMix.forward(x);
				}
				using(Tensor x = accumulate){
					using(y){
						accumulate = x.addcmul(y, finalGate, one);
					}
				}
				using (Tensor x = accumulate)
				{
					accumulate = x.add(bias);	
				}
				using (accumulate)
				{
					return CustomActivations.Norm(accumulate, epsilon).MoveToOuterDisposeScope();
				}
			}
		}
		private void FinalAccumulate(ref Tensor? accumulator, Tensor sliced){
			Tensor x = preFinalAccumulationGate.forward(sliced);
			using(Tensor z = x){
				x = z.sigmoid();
			}
			Tensor y = preFinalAccumulationInput.forward(sliced);
			using (Tensor z = y)
			{
				y = CustomActivations.HalfNorm(z);
			}

			using (Tensor z = y)
			{
				y = z.arctan();
			}
			using(x){
				using(y){
					if(accumulator is null){
						accumulator = x.mul(y);
					} else{
						using Tensor z = accumulator;
						accumulator = z.addcmul(x, z, one);
					}
				}
			}
		}
		public Tensor Decode(Tensor y){

			using (NewDisposeScope()){
				using (Tensor x = y)
				{
					y = x.mul(scale);
				}
				using (Tensor x = y)
				{
					y = x.matmul(wordEmbedding);
				}
				using(y){
					return y.add(finalBias).MoveToOuterDisposeScope();
				}

			}
		}
		private static readonly Scalar one = 1.0;
		public Tensor DefaultDecode(Tensor y){
			using (NewDisposeScope())
			{
				Tensor legacy;
				using(Tensor x = defaultEngine.forward(y)){
					legacy = Decode(x);
				}

				
				using (legacy)
				{
					using Tensor x2 = gigaDecoder.forward(y);
					return x2.maximum(legacy).MoveToOuterDisposeScope();
				}


			}
		}

		public Tensor Forward(ReadOnlySpan<ushort> input, int slice)
		{
			using(NewDisposeScope()){
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
			finalattention.L2Regularize(lambda);
			//Misc.L2RegularizeIMPL(preFinalAccumulationGate.weight, lambda);
			//Misc.L2RegularizeIMPL(preFinalAccumulationInput.weight, lambda);

			Misc.L2RegularizeIMPL(gigaDecoder.weight, lambda);
			foreach (Module layer in layers)
			{
				if (layer is IL2Regularizable regularizable)
				{
					regularizable.L2Regularize(lambda);
				}
			}
			//Misc.L2RegularizeIMPL(shortRangeAttnBoost.weight, lambda);
			//Misc.L2RegularizeIMPL(shortRangeAttnCompress.weight, lambda);
		}


	}


}

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





		private readonly ModuleList<Module<Tensor, Tensor>> prelayers = new ModuleList<Module<Tensor, Tensor>>();
		private readonly ModuleList<Module<Tensor, Tensor>> layers = new ModuleList<Module<Tensor, Tensor>>();
		//private readonly ModuleList<TinyRNN> rnnlayers = new ModuleList<TinyRNN>();
		//private readonly ModuleList<ResidualCausalConvolationalLookback> convAttentionLayers = new ModuleList<ResidualCausalConvolationalLookback>();
		private readonly Parameter wordEmbedding;
		private readonly MultiheadSelfAttention finalAttention;
		private readonly ResiduaComputeLayer finalCompute;


		private readonly int headcount;
		private readonly Scalar epsilon;

		private readonly int max_context_size;
		private readonly Parameter staticPositionalEncoding;
		private readonly Parameter classBias;
		public readonly Tensor supplementalWordEmbedding;
		//private readonly Linear premix;

		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int coreDepth, double positionalEncodingStd, int attentionValueSize, double epsilon, int max_context_size, int convAttentionSize, int convAttentionHeads, int convAttentionLayers, double wordEmbeddingStd, Tensor supplementalWordEmbedding, double initialAttentionGain, double initialConvAttentionGain, double initialComputeGain, double computeDropout) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}

			//this.pre_expand = Misc.CreateKaimingInitializedLinear(latentTokenSize, pre_expand, false, init.FanInOut.FanIn);
			//expand = Misc.CreateKaimingInitializedLinear(latentTokenSize, multipliedWidth, true, init.FanInOut.FanIn);

			finalCompute = new ResiduaComputeLayer("", latentTokenSize, epsilon, initialComputeGain, computeDropout);

			//engine = Misc.CreateXavierInitializedLinear(latentTokenSize, latentTokenSize, false);
			this.supplementalWordEmbedding = supplementalWordEmbedding;
			prelayers.Add(Misc.CreateXavierInitializedLinear(latentTokenSize + (int)supplementalWordEmbedding.size(1), latentTokenSize, true));

			prelayers.Add(LayerNorm(latentTokenSize, epsilon, false, false));
			for (int i = 0; i < convAttentionLayers; ++i)
			{
				prelayers.Add(new ResidualCausalConvolationalLookback("", latentTokenSize, convAttentionSize, convAttentionHeads, epsilon, initialConvAttentionGain));
				prelayers.Add(new ResiduaComputeLayer("", latentTokenSize, epsilon, initialComputeGain, computeDropout));
			}





			for (int i = 0; i < coreDepth; ++i)
			{
				layers.Add(new MultiheadSelfAttention("", latentTokenSize, attentionValueSize, attentionHeadsCount, epsilon, false, initialAttentionGain));
				layers.Add(new ResiduaComputeLayer("", latentTokenSize, epsilon, initialComputeGain, computeDropout));
			}



			//staticPositionalEncoding = Parameter(zeros(max_context_size, latentTokenSize));
			Span<long> wordEmbeddingSize = stackalloc long[2];


			
			wordEmbeddingSize[1] = latentTokenSize;

			if(positionalEncodingStd > 0.0){
				wordEmbeddingSize[0] = max_context_size;
				staticPositionalEncoding = Parameter(normal(0.0, positionalEncodingStd, wordEmbeddingSize));
			} else{
				staticPositionalEncoding = Parameter(zeros(max_context_size,latentTokenSize));
			}
			//staticPositionalEncoding = Parameter(randn(max_context_size, latentTokenSize));
			long ltk = supplementalWordEmbedding.size(0);

			wordEmbeddingSize[0] = ltk;





			//layers.Add(new TinyRNN("", multipliedWidth, multipliedWidth, epsilon));
			finalAttention = new MultiheadSelfAttention("", latentTokenSize, attentionValueSize, attentionHeadsCount, epsilon, false, initialAttentionGain);

			//finalBias = Parameter(zeros(1, pretrainedWord2Vec.size(0)));
			wordEmbedding = Parameter(normal(0.0, wordEmbeddingStd, wordEmbeddingSize));
			headcount = attentionHeadsCount;
			classBias = Parameter(zeros(ltk));
			RegisterComponents();
			this.epsilon = epsilon;
			this.max_context_size = max_context_size;
		}

		public Tensor Forward(ReadOnlySpan<ushort> input, int slice, double dropout = 0.0)
		{
			int len = input.Length;
			int maxlen2 = max_context_size;
			if(len > maxlen2 | len == 0)
			{
				throw new ArgumentOutOfRangeException(nameof(input));
			}
			

			long[] longs = new long[len];
			for(int i = 0; i < len; ++i){
				longs[i] = input[i];
			}

			using (NewDisposeScope())
			{

				



				Tensor[] all = new Tensor[len];
				Tensor wordEmbedding = this.wordEmbedding;
				Device device = wordEmbedding.device;


				Tensor y;
				Tensor spw;
				using(Tensor z2 = tensor(longs, ScalarType.Int64, device, false)){
					y = wordEmbedding[z2];
					spw = supplementalWordEmbedding[z2];
				}
				using(spw){
					using Tensor x = y;
					y = cat(new Tensor[]{spw, x},1);
				}




				//using (Tensor x = y)
				//y = premix.forward(x);

				foreach (Module<Tensor, Tensor> layer in prelayers)
				{
					using Tensor x = y;
					y = layer.forward(x);
				}
				
				using (Tensor x = y)
					if (len == maxlen2)
					{
						y = x.add(staticPositionalEncoding);
					}
					else
					{
						using Tensor z = staticPositionalEncoding.slice(0, 0, len, 1);
						y = x.add(z);
					}

				



				MultiheadSelfAttention finalattention = finalAttention;

				
				if(slice == 0 || layers.Count > 0) using (Tensor mask = Transformer.CreateCausalAttentionMask(len, len, ScalarType.Float32, device))
				{
					foreach (Module<Tensor, Tensor> hiddenLayer in layers)
					{
						using Tensor x = y;
						if (hiddenLayer is MultiheadSelfAttention multiheadSelfAttention)
						{
							y = multiheadSelfAttention.Forward(x, 0, mask, dropout);
						}
						else
						{
							y = hiddenLayer.forward(x);
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
				using (Tensor x = y)
					y = finalCompute.forward(x);
				/*
				if (slice > 0)
				{
					using Tensor x = y;
					y = x.slice(0, slice, len, 1);
				}
				*/

				using (Tensor x = y, x1 = wordEmbedding.transpose(0, 1))
					y = x.matmul(x1);

				using (y)
				{
					
					return classBias.add(y).MoveToOuterDisposeScope();
				}



				

			}
		}
		private static readonly Scalar one = 1.0;

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
			foreach (Module<Tensor, Tensor> module in prelayers)
			{
				if (module is IL2Regularizable regularizable)
				{
					regularizable.L2Regularize(lambda);
				}
			}
			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL2Regularizable regularizable)
				{
					regularizable.L2Regularize(lambda);
				}
			}
			finalAttention.L2Regularize(lambda);
			finalCompute.L2Regularize(lambda);
		}


	}


}
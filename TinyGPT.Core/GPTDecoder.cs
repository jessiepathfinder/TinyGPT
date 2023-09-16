using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Parameter = TorchSharp.Modules.Parameter;

namespace TinyGPT.Core
{
	
	public sealed class GPTDecoderV1 : Module<ReadOnlyMemory<Tensor>, Tensor>
	{
		public sealed class FullGPTDecoderUnitV1 : Module<ReadOnlyMemory<ushort>, Tensor>
		{
			private readonly ModuleList<BERTDictionaryItem> dictionaryItems;
			private readonly GPTDecoderV1 gptDecoder;

			public FullGPTDecoderUnitV1(string name, ModuleList<BERTDictionaryItem> dictionaryItems, GPTDecoderV1 gptDecoder) : base(name)
			{
				this.dictionaryItems = dictionaryItems ?? throw new ArgumentNullException(nameof(dictionaryItems));
				this.gptDecoder = gptDecoder ?? throw new ArgumentNullException(nameof(gptDecoder));
				RegisterComponents();
			}

			public override Tensor forward(ReadOnlyMemory<ushort> input)
			{
				return forward(input.Span);
			}
			public Tensor forward(ReadOnlySpan<ushort> input)
			{
				int len = input.Length;
				ModuleList<BERTDictionaryItem> cache = dictionaryItems;
				Tensor[] tensors = arrayPool.Rent(len);
				try
				{
					for(int i = 0; i < len; ++i){
						tensors[i] = cache[input[i]].parameters1;
					}
					return gptDecoder.forward(tensors.AsSpan(0, len));
				}
				finally {
					Misc.EraseReturnAsync(arrayPool, tensors, len);
				}
			}
		}
		//positional encoding
		private readonly Parameter positionWeights;
		private readonly Parameter positionBias;

		//transformer
		private readonly ModuleList<Trident> unmaskedAttention;
		private readonly ModuleList<Trident> maskedAttention;

		//predictor
		private readonly DenseStep denseStep1;
		private readonly DenseStep denseStep2;
		private readonly DenseStep denseStep3;
		private readonly DenseStep denseStep4;
		private readonly DenseStep denseStep5;
		private readonly Linear final;

		public GPTDecoderV1(ModuleList<Trident> unmaskedAttention, ModuleList<Trident> maskedAttention, int latentTokenSize, int tokenTypes, string name) : base(name)
		{

			this.unmaskedAttention = unmaskedAttention ?? throw new ArgumentNullException(nameof(unmaskedAttention));
			this.maskedAttention = maskedAttention ?? throw new ArgumentNullException(nameof(maskedAttention));
			CheckSafety();

			denseStep1 = new DenseStep(latentTokenSize, latentTokenSize, "");
			denseStep2 = new DenseStep(latentTokenSize, latentTokenSize, "");
			denseStep3 = new DenseStep(latentTokenSize, latentTokenSize, "");
			denseStep4 = new DenseStep(latentTokenSize, latentTokenSize, "");
			denseStep5 = new DenseStep(latentTokenSize, latentTokenSize, "");
			final = Linear(latentTokenSize, tokenTypes);

			positionBias = Parameter(randn(latentTokenSize));
			positionWeights = Parameter(randn(latentTokenSize));

			RegisterComponents();
		}
		public void CheckSafety(){
			if (unmaskedAttention.Count < 1)
			{
				throw new IndexOutOfRangeException("min 1 unmasked attention head required");
			}
			if (maskedAttention.Count < 1)
			{
				throw new IndexOutOfRangeException("min 1 masked attention head required");
			}
		}

		public override Tensor forward(ReadOnlyMemory<Tensor> input)
		{
			return forward(input.Span);
		}
		private static readonly ArrayPool<Tensor> arrayPool = ArrayPool<Tensor>.Create();
		public Tensor forward(ReadOnlySpan<Tensor> input)
		{
			CheckSafety();
			int len = input.Length;
			if(len == 0){
				throw new IndexOutOfRangeException("input can't be empty");
			}

			Tensor[] tensors = arrayPool.Rent(len * 3);
			Span<Tensor> ta = tensors.AsSpan(0, len);
			Span<Tensor> tb = tensors.AsSpan(len, len);
			Span<Tensor> tc = tensors.AsSpan(len * 2, len);
			Transformer.EncodePositionV2(input, ta, positionWeights, positionBias);

			try
			{
				ModuleList<Trident> temp = unmaskedAttention;
				for (int i = 0, heads = temp.Count; i < heads; ++i)
				{
					
					Trident trident = temp[i];
					if(i == 0){
						Transformer.ComputeSingleHeadAttention(trident, ta, tc);
					} else{
						Transformer.ComputeSingleHeadAttention(trident, ta, tb);
						for(int c = 0; c < len; ++c){
							tc[c] = tc[c].add(tb[c]);
						}
					}
				}
				temp = maskedAttention;
				Tensor? finalsum = null;
				for (int i = 0, heads = temp.Count; i < heads; ++i)
				{
					Tensor finalPosEncode = positionWeights.mul(len).add(positionWeights).cos();
					
					Trident trident = temp[i];

					Tensor aq;
					Tensor ak;
					{
						(aq, ak, Tensor av) = trident.forward(finalPosEncode);
						Tensor selfcode = Transformer.ScaledDotProductAttention(aq, ak, av);
						if (finalsum is null)
						{
							finalsum = selfcode;
						}
						else
						{
							finalsum = finalsum.add(selfcode);
						}
					}
					
					for (int c = 0; c < len; ++c){
						Tensor temp2 = tc[c];
						finalsum = finalsum.add(Transformer.ScaledDotProductAttention(aq, ak, trident.ValueOnly(temp2)));
					}
				}
				if(finalsum is null){
					throw new Exception("unexpected null final sum (should not reach here)");
				}

				return softmax(final.forward(denseStep5.forward(denseStep4.forward(denseStep3.forward(denseStep2.forward(denseStep1.forward(finalsum)))))), -1);
			} finally{
				Misc.EraseReturnAsync(arrayPool, tensors, len * 3);
			}
		}
	}
}

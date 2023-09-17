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
	public sealed class FullGPTDecoderUnitV1 : Module<ReadOnlyMemory<ushort>, Tensor>
	{
		private static readonly ArrayPool<Tensor> arrayPool = ArrayPool<Tensor>.Create();
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
			if(len == 0){
				throw new IndexOutOfRangeException("Input length cannot be zero");
			}
			ModuleList<BERTDictionaryItem> cache = dictionaryItems;
			Tensor[] tensors = arrayPool.Rent(len);
			try
			{
				for (int i = 0; i < len; ++i)
				{
					tensors[i] = cache[input[i]].parameters1;
				}
				return gptDecoder.forward(tensors.AsSpan(0, len));
			}
			finally
			{
				Misc.EraseReturnAsync(arrayPool, tensors, len);
			}
		}
	}
	public sealed class GPTDecoderV1 : Module<ReadOnlyMemory<Tensor>, Tensor>
	{
		
		
		//positional encoding
		private readonly Parameter positionWeights;
		private readonly Parameter positionBias;

		//transformer
		private readonly ModuleList<ModuleList<Trident>> transformerStages = new ModuleList<ModuleList<Trident>>();

		//predictor
		private readonly ModuleList<DenseStep> predictorStages = new ModuleList<DenseStep>();
		private readonly Linear final;

		public GPTDecoderV1(int transformerDepth, int attentionHeads, int predictorDepth, int latentTokenSize, int tokenTypes, string name) : base(name)
		{

			for(int i = 0; i < transformerDepth; ++i){
				ModuleList<Trident> trident = new ModuleList<Trident>();
				for(int z = 0; z < attentionHeads; ++z){
					trident.Add(new Trident("", latentTokenSize));
				}
				transformerStages.Add(trident);
			}
			for (int i = 0; i < predictorDepth; ++i)
			{
				predictorStages.Add(new DenseStep(latentTokenSize, latentTokenSize, ""));
			}


			final = Linear(latentTokenSize, tokenTypes);

			positionBias = Parameter(randn(latentTokenSize));
			positionWeights = Parameter(randn(latentTokenSize));

			RegisterComponents();
		}

		public override Tensor forward(ReadOnlyMemory<Tensor> input)
		{
			return forward(input.Span);
		}
		private static readonly long[] shape1 = { -1 };
		private static readonly long[] shape2 = { 1, -1 };
		public Tensor forward(ReadOnlySpan<Tensor> input)
		{
			int len = input.Length;
			if(len == 0){
				throw new IndexOutOfRangeException("input can't be empty");
			}
			Tensor[] ta = new Tensor[len];
			
			Transformer.EncodePositionV2(input, ta, positionWeights, positionBias);
			for (int i = 0; i < len; ++i)
			{
				ta[i] = input[i].reshape(shape2);
			}
			Tensor tensor = cat(ta, 0);
			foreach (ModuleList<Trident> tridents in transformerStages)
			{
				Tensor tensor1 = tensor;
				foreach(Trident trident in tridents) {
					(Tensor query, Tensor key, Tensor value) = trident.forward(tensor);
					tensor1 = tensor1.add(functional.scaled_dot_product_attention(query, key, value));
				}
				tensor = tensor1;
			}
			tensor = tensor[len - 1].reshape(shape1);
			foreach(DenseStep denseStep in predictorStages){
				tensor = denseStep.forward(tensor);
			}

			return final.forward(tensor);
		}
	}
}

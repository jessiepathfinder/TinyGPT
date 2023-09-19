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
	public abstract class FullGPTDecoderUnit : Module<ReadOnlyMemory<ushort>, Tensor>
	{
		protected FullGPTDecoderUnit(string name) : base(name)
		{
		}

		protected FullGPTDecoderUnit(nint handle, nint boxedHandle) : base(handle, boxedHandle)
		{
		}

		public abstract Tensor forward(ReadOnlySpan<ushort> input);
	}
	public sealed class FullGPTDecoderUnitV1 : FullGPTDecoderUnit
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
		public override Tensor forward(ReadOnlySpan<ushort> input)
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
		private readonly ModuleList<Tridentv2> attentionHeads = new ModuleList<Tridentv2>();
		private readonly ModuleList<DenseStepV2> attentionBypasses = new ModuleList<DenseStepV2>();

		//predictor
		private readonly ModuleList<Module<Tensor, Tensor>> predictorStages = new ModuleList<Module<Tensor, Tensor>>();

		public GPTDecoderV1(int attentionHeads1, int predictorDepth, int latentTokenSize, int tokenTypes, int attentionLatentSize, int predictorHiddenSize, int predictorFinalHiddenSize, string name) : base(name)
		{
			if (attentionHeads1 < 1) {
				throw new ArgumentOutOfRangeException(nameof(attentionHeads));
			}
			if (latentTokenSize < 1)
			{
				throw new ArgumentOutOfRangeException(nameof(latentTokenSize));
			}
			if (tokenTypes < 1)
			{
				throw new ArgumentOutOfRangeException(nameof(tokenTypes));
			}

			for (int i = 0; i < attentionHeads1; ++i){
				attentionHeads.Add(new Tridentv2("", latentTokenSize, attentionLatentSize));
				attentionBypasses.Add(new DenseStepV2(latentTokenSize, attentionLatentSize, ""));
			}
			int densewidth = attentionLatentSize * attentionHeads1;
			int prevsize = densewidth;

			for (int i = 0; i < predictorDepth; ++i)
			{
				predictorStages.Add(new DenseStepV2(prevsize, predictorHiddenSize, ""));
				prevsize = predictorHiddenSize;
			}
			predictorStages.Add(new DenseStepV2(prevsize, predictorFinalHiddenSize, false, ""));
			predictorStages.Add(Linear(predictorFinalHiddenSize, tokenTypes));

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
			int attentionCount = attentionHeads.Count;
			Tensor[] tb = new Tensor[attentionCount];
			int lm1 = len - 1;
			Tensor lasttensor = input[lm1].reshape(shape2);
			for (int i = 0; i < attentionCount; ++i) {
				(Tensor x, Tensor y, Tensor z) = attentionHeads[i].forward(tensor);
				tb[i] = functional.scaled_dot_product_attention(x, y, z)[lm1].reshape(shape2).add(attentionBypasses[i].forward(lasttensor)).reshape(shape1);
			}
			tensor = cat(tb, 0).reshape(shape2);
			foreach (Module<Tensor, Tensor> module in predictorStages){
				tensor = module.forward(tensor);
			}
			return softmax(tensor.reshape(shape1), -1);
		}
	}
}

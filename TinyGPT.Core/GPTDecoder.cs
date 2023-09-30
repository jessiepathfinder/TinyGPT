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
	public sealed class SimpleFullGPTDecoderUnit : FullGPTDecoderUnit{
		private readonly ModuleList<BERTDictionaryItem> dictionaryItems;
		private readonly GPTDecoderUnit decoder;
		private static readonly ArrayPool<Tensor> tensorpool = ArrayPool<Tensor>.Create();

		public SimpleFullGPTDecoderUnit(ModuleList<BERTDictionaryItem> dictionaryItems, GPTDecoderUnit decoder, string name) : base(name)
		{
			this.dictionaryItems = dictionaryItems ?? throw new ArgumentNullException(nameof(dictionaryItems));
			this.decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
			RegisterComponents();
		}

		public override Tensor Forward(ReadOnlySpan<ushort> input)
		{
			int len = input.Length;
			Tensor[] tensors = tensorpool.Rent(len);
			try{
				for(int i = 0; i < len; ++i){
					tensors[i] = dictionaryItems[input[i]].parameters1;
				}
				return decoder.Forward(tensors.AsSpan(0, len));
			} finally{
				Misc.EraseReturnAsync(tensorpool, tensors, len);
			}
			
		}
		public Tensor[] EncodeOnly(ReadOnlySpan<ushort> input)
		{
			int len = input.Length;
			Tensor[] tensors = new Tensor[len];
			try
			{
				for (int i = 0; i < len; ++i)
				{
					tensors[i] = dictionaryItems[input[i]].parameters1;
				}
				return tensors;
			}
			catch
			{
				for (int i = 0; i < len; ++i)
				{
					Tensor tensor = tensors[i];
					if(tensor is null){
						continue;
					}
					try{
						tensor.Dispose();
					} catch{
						
					}
				}
				throw;
			}

		}
	}
	public abstract class GPTDecoderUnit : Module<ReadOnlyMemory<Tensor>, Tensor>
	{
		protected GPTDecoderUnit(string name) : base(name)
		{
		}

		protected GPTDecoderUnit(nint handle, nint boxedHandle) : base(handle, boxedHandle)
		{
		}

		public abstract Tensor Forward(ReadOnlySpan<Tensor> input);
		public sealed override Tensor forward(ReadOnlyMemory<Tensor> input)
		{
			return Forward(input.Span);
		}
	}
	public sealed class SimpleAttentionHead : Module<ReadOnlyMemory<Tensor>, Tensor>
	{
		private readonly Linear keylayer;
		private readonly Linear valuelayer;
		private readonly Linear querylayer;
		private readonly Parameter positionalEncodingWeight;
		private readonly Parameter positionalEncodingBias;
		public SimpleAttentionHead(string name, int size) : base(name)
		{
			keylayer = Linear(size, size, false);
			valuelayer = Linear(size, size, false);
			querylayer = Linear(size, size, false);
			positionalEncodingWeight = Parameter(randn(size));
			positionalEncodingBias = Parameter(randn(size));
			RegisterComponents();
		}

		public override Tensor forward(ReadOnlyMemory<Tensor> input)
		{
			return Forward(input.Span);
		}
		private static readonly long[] shape2 = new long[] { 1, -1 };
		public Tensor Forward(ReadOnlySpan<Tensor> input)
		{
			int len = input.Length;
			if (len == 0)
			{
				throw new IndexOutOfRangeException(nameof(input));
			}
			using (NewDisposeScope())
			{
				Tensor y;
				using (NewDisposeScope())
				{
					Tensor[] tensors = new Tensor[len];
					for (int i = 0; i < len; ++i)
					{
						using (NewDisposeScope())
						{
							Tensor x = input[i].add(positionalEncodingWeight.mul(i).add(positionalEncodingBias).cos()).reshape(shape2);
							x.MoveToOuterDisposeScope();
							tensors[i] = x;
						}
					}
					y = cat(tensors, 0);
					y.MoveToOuterDisposeScope();
				}
				Tensor keys;
				Tensor values;
				Tensor queries;
				using(y){
					keys = keylayer.forward(y);
					values = valuelayer.forward(y);
					queries = querylayer.forward(y);
				}
				Tensor z = functional.scaled_dot_product_attention(queries, keys, values, is_casual: true);
				z.MoveToOuterDisposeScope();
				return z;
			}
		}


	}
	public sealed class GPTDecoderUnitV1 : GPTDecoderUnit
	{
		
		private readonly Linear finalLayer;
		private readonly Linear prefinal;
		private readonly ModuleList<SimpleAttentionHead> attentionHeads = new ModuleList<SimpleAttentionHead>();
		private readonly ModuleList<JessieNetLayer> jessienet = new ModuleList<JessieNetLayer>();

		private readonly int attentionHeadsCount;

		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int attentionFeedforwardDepth, int attentionFeedforwardHiddenSize, int tokenClasses, int prefinalhiddensize) : base(name)
		{
			if (attentionHeadsCount < 1){
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}

			int totalwidth = latentTokenSize * attentionHeadsCount;
			for (int i = 0; i < attentionHeadsCount; ++i){
				attentionHeads.Add(new SimpleAttentionHead("", latentTokenSize));
			}
			for (int i = 0; i < attentionFeedforwardDepth; ++i){
				jessienet.Add(new JessieNetLayer("", totalwidth, attentionFeedforwardHiddenSize));
			}
			finalLayer = Linear(prefinalhiddensize, tokenClasses);
			prefinal = Linear(totalwidth, prefinalhiddensize);
			this.attentionHeadsCount = attentionHeadsCount;
			RegisterComponents();
		}
		private static readonly long[] shape = new long[] { -1 };
		private static readonly long[] shape2 = new long[] { 1, -1 };
		public override Tensor Forward(ReadOnlySpan<Tensor> input)
		{
			using(Tensor x = Forward(input, input.Length - 1)){
				return x.reshape(shape);
			}
		}
		public Tensor Forward(ReadOnlySpan<Tensor> input, int slice)
		{
			int len = input.Length;
			if (len == 0)
			{
				throw new IndexOutOfRangeException(nameof(input));
			}

			using (NewDisposeScope()) {
				Tensor y;
				using (NewDisposeScope())
				{
					Tensor[] attentions = new Tensor[attentionHeadsCount];
					for (int i = 0; i < attentionHeadsCount; ++i)
					{
						using (Tensor x = attentionHeads[i].Forward(input))
						{
							attentions[i] = x.slice(0, slice, len, 1);
						}
					}
					y = cat(attentions, 1);
					y.MoveToOuterDisposeScope();
				}
				foreach(JessieNetLayer jessieNetLayer in jessienet){
					using Tensor z = y;
					y = jessieNetLayer.forward(z);
				}
				Tensor a;
				using(y){
					a = prefinal.forward(y);
				}
				using (a)
				{
					y = CustomActivations.LeakySoftplus(a);
				}
				using (y)
				{
					a = finalLayer.forward(y);
				}
				return a.softmax(1).MoveToOuterDisposeScope();
			}

			
		}

	}


}

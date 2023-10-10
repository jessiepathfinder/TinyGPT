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
	public sealed class SimpleFullGPTDecoderUnit : FullGPTDecoderUnit
	{
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
			try
			{
				for (int i = 0; i < len; ++i)
				{
					tensors[i] = dictionaryItems[input[i]].parameters1;
				}
				return decoder.Forward(tensors.AsSpan(0, len));
			}
			finally
			{
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
					if (tensor is null)
					{
						continue;
					}
					try
					{
						tensor.Dispose();
					}
					catch
					{

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
		private readonly Conv1d keylayer;
		private readonly Conv1d valuelayer;
		private readonly Conv1d querylayer;

		private readonly Parameter positionalEncodingWeight;
		private readonly Parameter positionalEncodingBias;
		private readonly double epsilon;
		private readonly ModuleList<JessieNetLayer> jessieNetLayers = new ModuleList<JessieNetLayer>();
		private readonly int lookback;

		public SimpleAttentionHead(string name, int size, int processorDepth, int processorHiddenSize, double epsilon, int convlookback) : base(name)
		{
			lookback = convlookback++;

			keylayer = Conv1d(size, size, convlookback, bias: false, padding: lookback);
			querylayer = Conv1d(size, size, convlookback, bias: false, padding: lookback);
			valuelayer = Conv1d(size, size, convlookback, bias: false, padding: convlookback);
			positionalEncodingWeight = Parameter(randn(size));
			positionalEncodingBias = Parameter(randn(size));

			this.epsilon = epsilon;
			for (int i = 0; i < processorDepth; ++i)
			{
				jessieNetLayers.Add(new JessieNetLayer("", size, processorHiddenSize, epsilon));
			}
			RegisterComponents();
		}

		public override Tensor forward(ReadOnlyMemory<Tensor> input)
		{
			return Forward(input.Span, 0);
		}
		private static readonly long[] shape2 = new long[] { 1, -1 };
		private static Tensor Truncate2(int start, int end, Tensor t)
		{
			using (t)
			{
				return t.slice(1, start, end, 1);
			}
		}
		private static Tensor Transpose2(Tensor t)
		{
			using (t)
			{
				return t.transpose(0, 1);
			}
		}
		public Tensor Forward(ReadOnlySpan<Tensor> input, int slice)
		{
			int len = input.Length;
			if (len == 0)
			{
				throw new IndexOutOfRangeException(nameof(input));
			}
			using (NewDisposeScope())
			{
				Tensor ry;
				Tensor y;
				using (NewDisposeScope())
				{
					Tensor[] tensors = new Tensor[len];
					for (int i = 0; i < len; ++i)
					{
						tensors[i] = input[i].reshape(shape2);
					}
					ry = cat(tensors, 0).MoveToOuterDisposeScope();
					for (int i = 0; i < len; ++i)
					{
						tensors[i].Dispose();
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
				Tensor z;
				int lookback = this.lookback;
				int finish = lookback + len;
				{
					using Tensor y2 = Transpose2(y);
					using Tensor keys = Transpose2(Truncate2(lookback, finish, keylayer.forward(y2)));

					using Tensor queries = Transpose2(Truncate2(lookback, finish, querylayer.forward(y2)));

					using Tensor tr2 = ry.transpose(0, 1);
					using Tensor values = Transpose2(Truncate2(0, len, valuelayer.forward(tr2)));
					using Tensor resvals = values.add(ry);
					z = functional.scaled_dot_product_attention(queries, keys, resvals, is_casual: true);
				}
				ry.Dispose();
				if (slice > 0)
				{
					using Tensor p = z;
					z = p.slice(0, slice, len, 1);
				}

				using (Tensor p = z)
				{
					z = CustomActivations.Norm(p, epsilon);
				}

				foreach (JessieNetLayer jessieNetLayer in jessieNetLayers)
				{
					using Tensor p = z;
					z = jessieNetLayer.forward(p);
				}
				z.MoveToOuterDisposeScope();
				return z;
			}
		}


	}
	public sealed class GPTDecoderUnitV1 : GPTDecoderUnit
	{

		private readonly Linear finalLayer;
		private readonly Linear viewCompressor;
		private readonly ModuleList<SimpleAttentionHead> attentionHeads = new ModuleList<SimpleAttentionHead>();
		private readonly ModuleList<JessieNetLayer> jessienet = new();

		private readonly int attentionHeadsCount;

		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int attentionFeedforwardDepth, int attentionFeedforwardHiddenSize, int tokenClasses, int compressedViewSize, int processorDepth, int processorHiddenSize, double epsilon, int convlookback) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}

			int totalwidth = latentTokenSize * attentionHeadsCount;
			for (int i = 0; i < attentionHeadsCount; ++i)
			{
				attentionHeads.Add(new SimpleAttentionHead("", latentTokenSize, processorDepth, processorHiddenSize, epsilon, convlookback));
			}
			for (int i = 0; i < attentionFeedforwardDepth; ++i)
			{
				jessienet.Add(new JessieNetLayer("", compressedViewSize, attentionFeedforwardHiddenSize, epsilon));
			}
			finalLayer = Linear(compressedViewSize, tokenClasses);
			viewCompressor = Linear(totalwidth, compressedViewSize);
			this.attentionHeadsCount = attentionHeadsCount;
			RegisterComponents();
		}
		private static readonly long[] shape = new long[] { -1 };
		private static readonly long[] shape2 = new long[] { 1, -1 };
		public override Tensor Forward(ReadOnlySpan<Tensor> input)
		{
			using Tensor x = Forward(input, input.Length - 1);
			return x[0];
		}
		public Tensor Forward(ReadOnlySpan<Tensor> input, int slice)
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
					Tensor[] attentions = new Tensor[attentionHeadsCount];
					for (int i = 0; i < attentionHeadsCount; ++i)
					{
						attentions[i] = attentionHeads[i].Forward(input, slice);
					}
					y = cat(attentions, 1);
					y.MoveToOuterDisposeScope();
				}
				using (Tensor z = y)
				{
					y = viewCompressor.forward(z);
				}
				using (Tensor z = y)
				{
					y = CustomActivations.LeakySoftplus(z);
				}
				foreach (JessieNetLayer jessieNetLayer in jessienet)
				{
					using Tensor z = y;
					y = jessieNetLayer.forward(z);
				}
				using (Tensor z = finalLayer.forward(y))
				{
					return z.softmax(1).MoveToOuterDisposeScope();
				}

			}


		}

	}


}

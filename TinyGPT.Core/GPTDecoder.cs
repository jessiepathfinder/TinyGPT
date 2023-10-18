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

		private readonly Parameter positionalEncodingWeight;
		private readonly Parameter positionalEncodingBias;
		private readonly ModuleList<JITNetLayer> JITLayers = new ModuleList<JITNetLayer>();

		public SimpleAttentionHead(string name, int size, int depth, double epsilon) : base(name)
		{

			positionalEncodingWeight = Parameter(randn(size));
			positionalEncodingBias = Parameter(randn(size));

			for (int i = 0; i < depth; ++i)
			{
				JITLayers.Add(new JITNetLayer("", size, epsilon));
			}
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
				foreach (JITNetLayer jitlayer in JITLayers)
				{
					using Tensor p = y;
					y = jitlayer.forward(p);
				}

				y.MoveToOuterDisposeScope();
				return y;
			}
		}


	}
	public sealed class GPTDecoderUnitV1 : GPTDecoderUnit
	{

		private readonly Linear finalLayer;
		private readonly Linear viewCompressor;
		private readonly ModuleList<SimpleAttentionHead> attentionHeads = new ModuleList<SimpleAttentionHead>();
		private readonly ModuleList<JITNetLayer> finaljit = new ModuleList<JITNetLayer>();

		private readonly int attentionHeadsCount;
		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int tokenClasses, int compressedViewSize, int firstTierAttentionDepth, int secondTierAttentionDepth, double epsilon) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}

			int totalwidth = latentTokenSize * attentionHeadsCount;
			for (int i = 0; i < attentionHeadsCount; ++i)
			{
				attentionHeads.Add(new SimpleAttentionHead("", latentTokenSize, firstTierAttentionDepth, epsilon));
			}
			for (int i = 0; i < secondTierAttentionDepth; ++i)
			{
				finaljit.Add(new JITNetLayer("", compressedViewSize, epsilon));
			}
			viewCompressor = Linear(totalwidth, compressedViewSize);
			finalLayer = Linear(compressedViewSize, tokenClasses);

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
						attentions[i] = attentionHeads[i].Forward(input);
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
				foreach (JITNetLayer jitlayer in finaljit)
				{
					using Tensor z = y;
					y = jitlayer.forward(z);
				}
				using (Tensor z = y)
				{
					y = z.slice(0, slice, len, 1);
				}
				using (y)
				{
					return finalLayer.forward(y).MoveToOuterDisposeScope();
				}
			}
		}

	}


}

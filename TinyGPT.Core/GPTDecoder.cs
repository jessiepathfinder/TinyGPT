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

	public sealed class SimpleAttentionHead : Module<ReadOnlyMemory<Tensor>, Tensor>
	{

		private readonly Parameter positionalEncodingWeight;
		private readonly Parameter positionalEncodingBias;
		private readonly ModuleList<AttentionBlock> attentionLayers = new ModuleList<AttentionBlock>();
		public void Regularize(double weight_l1_term, double bias_l2_term)
		{
			foreach(AttentionBlock attentionBlock in attentionLayers){
				attentionBlock.Regularize(weight_l1_term, bias_l2_term);
			}
		}

		public SimpleAttentionHead(string name, int size, int depth, double epsilon) : base(name)
		{

			positionalEncodingWeight = Parameter(randn(1, size));
			positionalEncodingBias = Parameter(randn(1, size));

			for (int i = 0; i < depth; ++i)
			{
				attentionLayers.Add(new AttentionBlock("", size, epsilon));
			}
			RegisterComponents();
		}

		public override Tensor forward(ReadOnlyMemory<Tensor> input)
		{
			return Forward(input.Span);
		}
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
							Tensor x = input[i].add(positionalEncodingWeight.mul(i).add(positionalEncodingBias).cos());
							x.MoveToOuterDisposeScope();
							tensors[i] = x;
						}
					}
					y = cat(tensors, 0);
					y.MoveToOuterDisposeScope();
				}
				foreach (AttentionBlock attentionBlock in attentionLayers)
				{
					using Tensor p = y;
					y = attentionBlock.forward(p);
				}

				y.MoveToOuterDisposeScope();
				return y;
			}
		}


	}
	public sealed class GPTDecoderUnitV1 : FullGPTDecoderUnit
	{
		private static readonly ArrayPool<Tensor> arrayPool = ArrayPool<Tensor>.Create();
		private readonly Linear finalLayer;
		private readonly ModuleList<SimpleAttentionHead> attentionHeads = new ModuleList<SimpleAttentionHead>();
		private readonly Linear wordEmbedding;
		public void Regularize(double weight_l1_term, double bias_l2_term)
		{
			foreach (SimpleAttentionHead attentionHead in attentionHeads)
			{
				attentionHead.Regularize(weight_l1_term, bias_l2_term);
			}
		}
		public Tensor ComputeKLDivergenceLoss(){
			Tensor wordweights = wordEmbedding.weight ?? throw new Exception("word embeddings does not have weight (should not reach here)");
			using(NewDisposeScope()){
				Tensor mean = wordweights.mean();
				
				Tensor y;
				using(Tensor x = wordweights.sub(mean)){
					y = x.square();
				}
				Tensor stddev_square;
				using(y){
					stddev_square = y.mean();
				}

				return stddev_square.add(mean.square()).sub(stddev_square.log()).sub(1).MoveToOuterDisposeScope();
			}
		}

		private readonly int attentionHeadsCount;
		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int tokenClasses, int firstTierAttentionDepth, double epsilon) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}

			for (int i = 0; i < attentionHeadsCount; ++i)
			{
				attentionHeads.Add(new SimpleAttentionHead("", latentTokenSize, firstTierAttentionDepth, epsilon));
			}
			finalLayer = Linear(latentTokenSize * attentionHeadsCount, latentTokenSize);
			wordEmbedding = Linear(latentTokenSize, tokenClasses, false);
			this.attentionHeadsCount = attentionHeadsCount;
			RegisterComponents();
		}

		public Tensor Forward(ReadOnlySpan<ushort> input, int slice)
		{
			int len = input.Length;
			if (len == 0)
			{
				throw new IndexOutOfRangeException(nameof(input));
			}
			Tensor wordweights = wordEmbedding.weight ?? throw new Exception("word embeddings does not have weight (should not reach here)");
			using (NewDisposeScope())
			{
				Tensor y;
				using (NewDisposeScope())
				{
					Tensor[] attentions = new Tensor[attentionHeadsCount];

					Tensor[] borrowed = arrayPool.Rent(len);
					try{
						using DisposeScope disposeScope = NewDisposeScope();
						for(int i = 0; i < len; ++i){
							borrowed[i] = wordweights[input[i]];
						}
						ReadOnlySpan<Tensor> span = borrowed.AsSpan(0, len);
						for (int i = 0; i < attentionHeadsCount; ++i)
						{
							attentions[i] = attentionHeads[i].Forward(span).MoveToOuterDisposeScope();
						}
					} finally{
						Misc.EraseReturnAsync(arrayPool, borrowed, len);
					}
					y = cat(attentions, 1);
					y.MoveToOuterDisposeScope();
				}
				using(Tensor x = y){
					y = x.slice(0, slice, len, 1);
				}
				using (Tensor x = y)
				{
					y = finalLayer.forward(x);
				}
				using(y){
					return wordEmbedding.forward(y).MoveToOuterDisposeScope();
				}
			}
		}

		public override Tensor Forward(ReadOnlySpan<ushort> input)
		{
			return Forward(input, 0);
		}
	}


}

using System;
using System.Buffers;
using System.Collections;
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
	public abstract class FullGPTDecoderUnit : Module
	{
		protected FullGPTDecoderUnit(string name) : base(name)
		{
		}


		public abstract Tensor Encode(ReadOnlySpan<ushort> input, Tensor? memory);
		public abstract Tensor Decode(Tensor state, int slice, Tensor? memory);
	}


	public sealed class GPTDecoderUnitV1 : FullGPTDecoderUnit
	{
		private sealed class GPTFinalAttentionLayer : Module<Tensor, Tensor, Tensor, Tensor?, Tensor>
		{
			private readonly int headcount;
			private readonly int latentTokenSize;
			private readonly Parameter residual;
			private readonly LayerNorm layerNorm;
			private readonly Linear tail;
			private readonly ModuleList<AttentionLayer> attentionHeads = new ModuleList<AttentionLayer>();

			public GPTFinalAttentionLayer(int latentTokenSize, int heads, double epsilon) : base("")
			{
				int size = latentTokenSize * heads;
				headcount = heads;
				this.latentTokenSize = latentTokenSize;
				for (int i = 0; i < heads; ++i)
				{
					attentionHeads.Add(new AttentionLayer("", latentTokenSize));
				}
				residual = Parameter(ones(1, size));
				layerNorm = LayerNorm(size, epsilon, false);
				tail = Misc.CreateXavierInitializedLinear(size, size, true);
				RegisterComponents();
			}

			public sealed override Tensor forward(Tensor a, Tensor b, Tensor attmask, Tensor? memory)
			{
				using (NewDisposeScope())
				{
					int heads = headcount;
					int latentTokenSize = this.latentTokenSize;
					Tensor x;
					Tensor[] attnheads;
					Tensor[] attnheads2 = b.split(latentTokenSize, 1);
					if(memory is null){
						attnheads = a.split(latentTokenSize, 1);
					} else{
						using Tensor y = cat(new Tensor[] { memory, a }, 0);
						attnheads = y.split(latentTokenSize, 1);
					}
					using (NewDisposeScope())
					{
						for (int i = 0; i < heads; ++i)
						{
							using Tensor y = attnheads[i];
							using Tensor y2 = attnheads2[i];
							attnheads[i] = attentionHeads[i].Forward(y, y2, attmask);
						}
						x = cat(attnheads, 1).MoveToOuterDisposeScope();
					}


					using (Tensor y = x)
					{
						x = tail.forward(y);
					}
					Tensor z;
					using (Tensor y = b.mul(residual))
					{
						using (x)
						{
							z = y.add(x);
						}
					}
					using (z)
					{
						return layerNorm.forward(z).MoveToOuterDisposeScope();
					}

				}
			}
		}
		private sealed class GPTHiddenLayer : Module<Tensor, Tensor, Tensor?, Tensor>
		{
			private readonly Parameter residual;
			private readonly LayerNorm layerNorm;
			private readonly Linear tail;
			private readonly Module<Tensor, Tensor, Tensor?, Tensor> module;
			public GPTHiddenLayer(Module<Tensor, Tensor, Tensor?, Tensor> module, int size, double epsilon) : base("")
			{
				residual = Parameter(ones(1, size));
				layerNorm = LayerNorm(size, epsilon, false);
				tail = Misc.CreateXavierInitializedLinear(size, size, true);
				this.module = module;
				RegisterComponents();
			}

			public sealed override Tensor forward(Tensor input, Tensor attmask, Tensor? memory)
			{
				using (NewDisposeScope())
				{
					Tensor x;
					using (Tensor y = module.forward(input, attmask, memory))
					{
						x = tail.forward(y);
					}
					Tensor z;
					using (Tensor y = input.mul(residual))
					{
						using (x)
						{
							z = y.add(x);
						}
					}
					using (z)
					{
						return layerNorm.forward(z).MoveToOuterDisposeScope();
					}

				}
			}
		}
		private sealed class GPTAttentionLayer : Module<Tensor, Tensor, Tensor?, Tensor>
		{
			private readonly int headcount;
			private readonly int latentTokenSize;
			private readonly ModuleList<AttentionLayer> attentionHeads = new ModuleList<AttentionLayer>();
			public GPTAttentionLayer(int latentTokenSize, int heads) : base("")
			{
				headcount = heads;
				this.latentTokenSize = latentTokenSize;
				for (int i = 0; i < heads; ++i)
				{
					attentionHeads.Add(new AttentionLayer("", latentTokenSize));
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor input, Tensor attmask, Tensor? memory)
			{
				int heads = headcount;
				int latentTokenSize = this.latentTokenSize;
				Tensor[] attnheads = input.split(latentTokenSize, 1);
				if (memory is null){
					for (int i = 0; i < heads; ++i)
					{
						using Tensor x = attnheads[i];
						attnheads[i] = attentionHeads[i].Forward(x, attmask);
					}
					try
					{
						return cat(attnheads, 1);
					}
					finally
					{
						for (int i = 0; i < heads; ++i)
						{
							attnheads[i].Dispose();
						}
					}
				}
				Tensor[] attnheads2;
				using (Tensor att2 = cat(new Tensor[] { memory, input })){
					attnheads2 = att2.split(latentTokenSize, 1);
				}
					
				for (int i = 0; i < heads; ++i)
				{
					using Tensor y = attnheads[i];
					using Tensor y2 = attnheads2[i];
					attnheads[i] = attentionHeads[i].Forward(y2, y, attmask);
				}
				return cat(attnheads, 1).MoveToOuterDisposeScope();

			}
		}
		private sealed class GPTComputeLayer : Module<Tensor, Tensor, Tensor?, Tensor>
		{
			private readonly Linear linear;
			public GPTComputeLayer(int size) : base("")
			{
				linear = Misc.CreateXavierInitializedLinear(size, size, true);
				RegisterComponents();
			}

			public override Tensor forward(Tensor input, Tensor attmask, Tensor? memory)
			{
				using (Tensor x = linear.forward(input)){
					return x.gelu().MoveToOuterDisposeScope();
				}
				

			}
		}




		private readonly Linear outputEmbedding;
		private readonly ModuleList<GPTHiddenLayer> layers = new ModuleList<GPTHiddenLayer>();
		private readonly Tensor wordEmbedding;


		private readonly double scale;



		private readonly Parameter positionalEncodingBias;
		private readonly Parameter positionalEncodingWeight;
		private readonly GPTFinalAttentionLayer finalAttentionLayer;
		private readonly int headcount;
		//private readonly LayerNorm layerNorm;
		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int tokenClasses, int coreDepth, double initialFrequency, double epsilon) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}
			int fullsize = latentTokenSize * attentionHeadsCount;
			positionalEncodingWeight = Parameter(randn(fullsize, 1).mul_(initialFrequency));
			positionalEncodingBias = Parameter(zeros(fullsize, 1));

			for (int i = 0; i < coreDepth; ++i)
			{
				layers.Add(new GPTHiddenLayer(new GPTAttentionLayer(latentTokenSize, attentionHeadsCount), fullsize, epsilon));
				layers.Add(new GPTHiddenLayer(new GPTComputeLayer(fullsize), fullsize, epsilon));
			}


			outputEmbedding = Misc.CreateXavierInitializedLinear(fullsize, latentTokenSize, true);
			finalAttentionLayer = new GPTFinalAttentionLayer(latentTokenSize, attentionHeadsCount, epsilon);
			wordEmbedding = randn(latentTokenSize, tokenClasses);
			scale = Math.Sqrt(latentTokenSize);
			headcount = attentionHeadsCount;
			//layerNorm = LayerNorm(tokenClasses, epsilon);
			RegisterComponents();
		}
		public override Tensor Decode(Tensor state, int slice, Tensor? memory)
		{
			int len = (int)state.size(0);
			int attsize = memory is null ? len : (len + (int)memory.size(0));
			Tensor y;
			if(slice > 0){
				using Tensor query = state.slice(0, slice, len, 1);
				using Tensor mask2 = Transformer.CreateCausalAttentionMask(len - slice, attsize, wordEmbedding.dtype, wordEmbedding.device);
				y = finalAttentionLayer.forward(state, query, mask2, memory);
			} else{
				using Tensor mask2 = Transformer.CreateCausalAttentionMask(len, attsize, wordEmbedding.dtype, wordEmbedding.device);
				y = finalAttentionLayer.forward(state, state, mask2, memory);
			}
			using (Tensor x = y)
			{
				y = outputEmbedding.forward(x);
			}

			using (Tensor x = y)
			{
				y = x.matmul(wordEmbedding);
			}
			using (y)
			{
				return y.div(scale).MoveToOuterDisposeScope();
			}

		}
		public override Tensor Encode(ReadOnlySpan<ushort> input, Tensor? memory)
		{
			int len = input.Length;
			if (len == 0)
			{
				throw new IndexOutOfRangeException(nameof(input));
			}
			using (NewDisposeScope())
			{

				int headcount = this.headcount;
				Tensor[] all = new Tensor[len];
				Tensor[] heads2 = new Tensor[headcount];
				Tensor wordEmbedding = this.wordEmbedding;
				for (int i = 0; i < len; ++i)
				{
					long my = input[i];
					Tensor n;
					using (Tensor p = positionalEncodingWeight.mul(i))
					{
						n = p.add(positionalEncodingBias);
					}
					using (Tensor p = n)
					{
						n = p.sin();
					}

					Tensor z2;
					using (Tensor z = wordEmbedding.slice(1, my, my + 1, 1))
					{
						for (int p = 0; p < headcount; ++p)
						{
							heads2[p] = z;
						}
						z2 = cat(heads2, 0);
					}
					using (z2)
					{
						using (n)
						{
							all[i] = z2.add(n);
						}
					}

				}

				Tensor y;
				using (Tensor c2 = cat(all, 1))
				{
					y = c2.transpose(0, 1);
				}

				using (Tensor attentionMask = Transformer.CreateCausalAttentionMask(len, memory is null ? len : (len + (int)memory.size(0)), wordEmbedding.dtype, wordEmbedding.device))
				{
					foreach (GPTHiddenLayer hiddenLayer in layers)
					{
						using Tensor x = y;
						y = hiddenLayer.forward(x, attentionMask, memory);
					}
					
				}
				return y.MoveToOuterDisposeScope();
			}
		}
	}


}

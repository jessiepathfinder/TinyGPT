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


	public sealed class GPTDecoderUnitV1 : FullGPTDecoderUnit, IL1Regularizable
	{



		


		private readonly ModuleList<Module<Tensor, Tensor>> layers = new ModuleList<Module<Tensor, Tensor>>();
		private readonly Tensor wordEmbedding;


		private readonly double scale;



		private readonly Parameter positionalEncodingBias;
		private readonly Parameter positionalEncodingWeight;
		private readonly int headcount;
		//private readonly LayerNorm layerNorm;
		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int tokenClasses, int coreDepth, double initialFrequency, int attentionKeySize, int attentionValueSize, double epsilon) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}
			positionalEncodingWeight = Parameter(randn(latentTokenSize, 1).mul_(initialFrequency));
			positionalEncodingBias = Parameter(zeros(latentTokenSize, 1));

			for (int i = 0; i < coreDepth; ++i)
			{
				layers.Add(new MultiheadResidualAttention("", latentTokenSize, attentionKeySize, attentionValueSize, latentTokenSize, attentionHeadsCount, epsilon));
				layers.Add(new ResidualGatedComputeLayer("", latentTokenSize, epsilon));
			}


			wordEmbedding = randn(latentTokenSize, tokenClasses);
			scale = Math.Sqrt(latentTokenSize * 2);
			headcount = attentionHeadsCount;
			//layerNorm = LayerNorm(tokenClasses, epsilon);
			RegisterComponents();
		}

		public Tensor Forward(ReadOnlySpan<ushort> input, int slice)
		{
			int len = input.Length;
			if (len == 0)
			{
				throw new IndexOutOfRangeException(nameof(input));
			}
			using (NewDisposeScope())
			{

				int headcount = this.headcount;

				Tensor wordEmbedding = this.wordEmbedding;
				Tensor positionalEncoding;
				using(Tensor range = arange(0, len, wordEmbedding.dtype, wordEmbedding.device)){
					positionalEncoding = range.mul(positionalEncodingWeight);
				}
				using(Tensor tempp = positionalEncoding){
					positionalEncoding = tempp.add(positionalEncodingBias);
				}
				using (Tensor tempp = positionalEncoding)
				{
					positionalEncoding = tempp.sin();
				}

				Tensor[] all = new Tensor[len];
				

				Tensor y;
				using(NewDisposeScope()){
					for (int i = 0; i < len; ++i)
					{
						long my = input[i];
						all[i] = wordEmbedding.slice(1, my, my + 1, 1);
					}
					y = cat(all, 1).MoveToOuterDisposeScope();
				}
				using (Tensor c2 = y)
				{
					using(positionalEncoding){
						y = c2.add(positionalEncoding);
					}
				}
				using (Tensor c2 = y)
				{
					y = c2.transpose(0, 1);
				}

				using (Tensor mask = Transformer.CreateCausalAttentionMask(len, len, wordEmbedding.dtype, wordEmbedding.device)){
					foreach (Module<Tensor, Tensor> hiddenLayer in layers)
					{
						using Tensor x = y;
						if (hiddenLayer is MultiheadResidualAttention multiheadResidualAttention)
						{
							y = multiheadResidualAttention.Forward(y, y, mask);
						}
						else
						{
							y = hiddenLayer.forward(x);
						}
					}
				}
				

				using (Tensor x = y)
				{
					y = x.slice(0, slice, len, 1);
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
		}

		public override Tensor Forward(ReadOnlySpan<ushort> input)
		{
			using (NewDisposeScope())
			{
				using Tensor x = Forward(input, input.Length - 1);
				return x.squeeze(0).MoveToOuterDisposeScope();
			}
		}

		public void L1Regularize(double lambda)
		{
			foreach(Module<Tensor, Tensor> module in layers){
				if(module is IL1Regularizable regularizable){
					regularizable.L1Regularize(lambda);
				}
			}
		}
	}


}
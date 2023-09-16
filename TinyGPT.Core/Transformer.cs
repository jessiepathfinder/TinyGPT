using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;
using static TorchSharp.torch.nn;
using System.Buffers;

namespace TinyGPT.Core
{
	public sealed class BERTDictionaryItem : Module<Tensor, Tensor>
	{
		public readonly Parameter parameters1;
		public BERTDictionaryItem(string name, long size) : base(name) {
			parameters1 = Parameter(randn(size), true);
			RegisterComponents();
		}

		public override Tensor forward(Tensor y)
		{
			return parameters1.add(y);
		}
	}
	public class Trident : Module<Tensor, (Tensor, Tensor, Tensor)>
	{
		private readonly Linear linear1;
		private readonly Linear linear2;
		private readonly Linear linear3;

		public Trident(string name, int size) : base(name)
		{
			linear1 = Linear(size, size, false);
			linear2 = Linear(size, size, false);
			linear3 = Linear(size, size, false);
			RegisterComponents();
		}

		public override (Tensor, Tensor, Tensor) forward(Tensor input1)
		{
			return (linear1.forward(input1), linear2.forward(input1), linear3.forward(input1));
		}
		public Tensor ValueOnly(Tensor input){
			return linear3.forward(input);
		}
	}
	


	public static class Transformer{
		public static Tensor PositionalEncodingV2(Tensor input, Tensor weights, Tensor biases, int index)
		{
			return input.add(weights.mul(index).add(biases).sin());
		}
		public static void EncodeWord2Vec(ModuleList<BERTDictionaryItem> list, ReadOnlySpan<ushort> span, Span<Tensor> outputs) {
			int len = span.Length;
			if (outputs.Length < len) {
				throw new ArgumentOutOfRangeException("output can't be smaller than input");
			}
			for (int ctr = 0; ctr < len; ++ctr)
			{
				outputs[ctr] = list[span[ctr] + 2].parameters1;
			}
		}
		public static void EncodePositionV2(ReadOnlySpan<Tensor> span, Span<Tensor> outputs, Tensor weights, Tensor biases)
		{
			int len = span.Length;
			if (outputs.Length < len)
			{
				throw new ArgumentOutOfRangeException("output can't be smaller than input");
			}
			for (int ctr = 0; ctr < len; ++ctr)
			{
				outputs[ctr] = PositionalEncodingV2(span[ctr], weights, biases, ctr);
			}
		}

		public static int Tokenize(IReadOnlyDictionary<string, ushort> dict, Span<ushort> output, string str, int maxtokensize)
		{
			if(maxtokensize < 1){
				throw new ArgumentOutOfRangeException(nameof(maxtokensize));
			}
			int pos = 0;
			for (int ctr = 0, len = str.Length, outlen = output.Length; ctr < len & pos < outlen;)
			{
				StringBuilder sb = new StringBuilder();
				int token = -1;
				for (int i = ++ctr, stop = Math.Min(i + maxtokensize, len); i < stop; ++i)
				{
					sb.Append(str[i]);
					if (dict.TryGetValue(sb.ToString(), out ushort val))
					{
						token = val;
						ctr = i;
					}
				}
				if (token > -1)
				{
					output[pos++] = (ushort)(token + 2);
				}
			}
			return pos;
		}
		private static readonly ArrayPool<(Tensor, Tensor, Tensor)> arrayPool1 = ArrayPool<(Tensor, Tensor, Tensor)>.Create();
		public static void ComputeSingleHeadAttention(Trident trident, ReadOnlySpan<Tensor> inputs, Span<Tensor> outputs)
		{
			int limit = inputs.Length;
			if(limit == 0){
				return;
			}
			if(outputs.Length < limit) {
				throw new IndexOutOfRangeException("outputs cannot be smaller than input values");
			}
			(Tensor, Tensor, Tensor)[] tmpcache = arrayPool1.Rent(limit);
			for(int i = 0; i < limit; ++i){
				tmpcache[i] = trident.forward(inputs[i]);
			}
			try{
				
				for (int i = 0; i < limit; ++i)
				{
					(Tensor query, Tensor key, _) = tmpcache[i];
					Tensor z = inputs[i];
					for (int c = 0; c < limit; ++c)
					{
						(_, _, Tensor value) = tmpcache[c];
						Tensor res = ScaledDotProductAttention(query, key, value);
						z = z.add(res);
					}
					outputs[i] = z;
				}
			} finally{
				ThreadPool.QueueUserWorkItem((((Tensor, Tensor, Tensor)[] tmpcache, int limit1) x) =>
				{
					for(int i = 0;i < x.limit1; ++i){
#pragma warning disable CS8619 // Nullability of reference types in value doesn't match target type.
						x.tmpcache[i] = (null, null, null);
#pragma warning restore CS8619 // Nullability of reference types in value doesn't match target type.
					}
					arrayPool1.Return(x.tmpcache, false);
				}, (tmpcache, limit), true);
			}
		}
		private static readonly long[] shape1 = { -1 };
		private static readonly long[] shape2 = { -1, 1 };


		public static Tensor ScaledDotProductAttention(Tensor query, Tensor key, Tensor value)
		{
			return value.mul(query.dot(key).div(MathF.Sqrt(query.size(0))).sigmoid());
		}


	}
}

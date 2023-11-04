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
using System.Collections;

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


		public static int Tokenize(IReadOnlyDictionary<string, ushort> dict, Span<ushort> output, string str, int maxtokensize, int specialTokenClasses)
		{
			if(maxtokensize < 1){
				throw new ArgumentOutOfRangeException(nameof(maxtokensize));
			}
			int pos = 0;
			for (int ctr = 0, len = str.Length, outlen = output.Length; ctr < len & pos < outlen;)
			{
				StringBuilder sb = new StringBuilder();
				int token = -1;
				for (int i = ctr++, stop = Math.Min(i + maxtokensize, len); i < stop; ++i)
				{
					sb.Append(str[i]);
					if (dict.TryGetValue(sb.ToString(), out ushort val))
					{
						token = val;
						ctr = i + 1;
					}
				}
				if (token > -1)
				{
					output[pos++] = (ushort)(token + specialTokenClasses);
				}
			}
			return pos;
		}
		private static readonly long[] shape1 = { -1 };
		private static readonly long[] shape2 = { 1, -1 };


	}
}

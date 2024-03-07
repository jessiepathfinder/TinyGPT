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
using System.Security.Cryptography;

namespace TinyGPT.Core
{
	public sealed class BERTDictionaryItem : Module<Tensor, Tensor>
	{
		public readonly Parameter parameters1;
		public BERTDictionaryItem(string name, long size) : base(name)
		{
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
		public Tensor ValueOnly(Tensor input)
		{
			return linear3.forward(input);
		}
	}





	public static class Transformer
	{


		public static int Tokenize(IReadOnlyDictionary<string, OptimizedTokenizerEntry> dict, Span<ushort> output, ReadOnlySpan<char> str, int maxtokensize, int specialTokenClasses)
		{
			if (maxtokensize < 1)
			{
				throw new ArgumentOutOfRangeException(nameof(maxtokensize));
			}
			int pos = 0;
			int ctr2 = 0;
			for (int len = str.Length, outlen = output.Length; ctr2 < len & pos < outlen;)
			{
				StringBuilder sb = new StringBuilder();
				int token = -1;
				for (int i = ctr2++, stop = Math.Min(i + maxtokensize, len); i < stop; ++i)
				{
					sb.Append(str[i]);
					if (dict.TryGetValue(sb.ToString(), out OptimizedTokenizerEntry val))
					{
						token = val.value;
						ctr2 = i + 1;
						if(val.fastret){
							break;
						}
					}
				}
				if (token > -1)
				{
					output[pos++] = (ushort)(token + specialTokenClasses);
				}
			}
			return pos;
		}
		public static Tensor CreateSemiCausalAttentionMask(int size, int startCausal, ScalarType scalarType, Device? device)
		{
			using (NewDisposeScope())
			{
				//[query, key]
				Tensor x = zeros(size - startCausal, size - startCausal, scalarType, device);





				using (Tensor y = ones(size - startCausal, size - startCausal, ScalarType.Bool, device))
				{
					y.tril_(0);
					y.logical_not_();
					x.masked_fill_(y, double.NegativeInfinity);
				}
				Tensor[] arr;
				using (Tensor b = zeros(startCausal, size - startCausal, scalarType, device))
				{
					b.fill_(double.NegativeInfinity);
					arr = new Tensor[] { b, x };
					using (x)
					{
						arr[1] = cat(arr, 0);
					}
				}
				using (Tensor b = zeros(size, startCausal, scalarType, device))
				{
					arr[0] = b;
					return cat(arr, 1).MoveToOuterDisposeScope();
				}
			}
		}

		public static Tensor CreateCausalAttentionMask(int querySize, int keysize, ScalarType scalarType, Device? device)
		{
			using (NewDisposeScope())
			{
				Tensor x = zeros(querySize, keysize, scalarType, device);
				using (Tensor y = ones(querySize, keysize, ScalarType.Bool, device))
				{
					y.tril_(keysize - querySize);
					y.logical_not_();
					x.masked_fill_(y, double.NegativeInfinity);
				}

				return x.MoveToOuterDisposeScope();
			}
		}

		public static void Mask(ReadOnlySpan<ushort> tokens, Span<ushort> outputs, byte maskProbability, ushort maskToken, Dictionary<ushort, bool>? state)
		{
			int len = tokens.Length;
			if (outputs.Length < len)
			{
				throw new IndexOutOfRangeException(nameof(outputs));
			}
			int randsegment = Math.Min(len, 1024);

			Span<byte> actions = stackalloc byte[randsegment];


			if(state is null){
				state = new Dictionary<ushort, bool>();
			}
			for (int i = 0; i < len; ++i)
			{
				int imod = i % 1024;
				if (imod == 0)
				{
					int remains = len - i;
					RandomNumberGenerator.Fill(remains < randsegment ? actions[..remains] : actions);
				}
				ushort current = tokens[i];
				outputs[i] = (state.TryAdd(current, false) || (actions[imod] > maskProbability)) ? current : maskToken;
			}
		}
		public static int MaskOrRamdomRemove(ReadOnlySpan<ushort> tokens, Span<ushort> outputs, byte maskProbability, byte randomRemoveProbability, ushort maskToken, Dictionary<ushort, bool>? state)
		{
			int len = tokens.Length;
			if (outputs.Length < len)
			{
				throw new IndexOutOfRangeException(nameof(outputs));
			}
			int randsegment = Math.Min(len, 1024);

			Span<byte> actions = stackalloc byte[randsegment];


			if (state is null)
			{
				state = new Dictionary<ushort, bool>();
			}
			int p2 = 0;
			for (int i = 0; i < len; ++i)
			{
				int imod = i % 1024;
				if (imod == 0)
				{
					int remains = len - i;
					RandomNumberGenerator.Fill(remains < randsegment ? actions[..remains] : actions);
				}
				ushort current = tokens[i];
				if(!state.TryAdd(current, false)){
					byte action = actions[imod];
					if(action <= randomRemoveProbability){
						//skip token
						continue;
					}
					if(action <= maskProbability){
						current = maskToken;
					}
				}

				outputs[p2++] = current;
			}
			return p2;
		}
		public static int RamdomRemove(ReadOnlySpan<ushort> tokens, Span<ushort> outputs, byte randomRemoveProbability, Dictionary<ushort, bool>? state)
		{
			int len = tokens.Length;
			if (outputs.Length < len)
			{
				throw new IndexOutOfRangeException(nameof(outputs));
			}
			int randsegment = Math.Min(len, 1024);

			Span<byte> actions = stackalloc byte[randsegment];


			if (state is null)
			{
				state = new Dictionary<ushort, bool>();
			}
			int p2 = 0;
			for (int i = 0; i < len; ++i)
			{
				int imod = i % 1024;
				if (imod == 0)
				{
					int remains = len - i;
					RandomNumberGenerator.Fill(remains < randsegment ? actions[..remains] : actions);
				}
				ushort current = tokens[i];
				if (!state.TryAdd(current, false))
				{
					byte action = actions[imod];
					if (action <= randomRemoveProbability)
					{
						//skip token
						continue;
					}
				}

				outputs[p2++] = current;
			}
			return p2;
		}

		private static readonly long[] shape1 = { -1 };
		private static readonly long[] shape2 = { 1, -1 };


	}
}

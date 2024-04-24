using Google.Protobuf.WellKnownTypes;
using ICSharpCode.SharpZipLib.BZip2;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TinyGPT.Core
{
	public readonly struct OptimizedTokenizerEntry{
		public readonly ushort value;
		public readonly bool fastret;

		public OptimizedTokenizerEntry(ushort value, bool fastret)
		{
			this.value = value;
			this.fastret = fastret;
		}
	}
	public static class Misc
	{
		public static Dictionary<string, OptimizedTokenizerEntry> OptimizeDictionary(IReadOnlyDictionary<string, ushort> input){
			string[] keys = input.Keys.ToArray();
			int len = keys.Length;
			Dictionary<string, OptimizedTokenizerEntry> thedict = new Dictionary<string, OptimizedTokenizerEntry>(len);

			foreach (KeyValuePair<string, ushort> kvp in input){
				bool fastret = true;
				string str = kvp.Key;

				for(int i = 0, sl = str.Length; i < len; ){
					string str2 = keys[i++];
					if (str2.Length > sl && str2.StartsWith(str)){
						fastret = false;
						break;
					}
				}
				thedict.Add(str, new OptimizedTokenizerEntry(kvp.Value, fastret));
			}
			return thedict;
		}
		private static readonly NLLLoss nlloss = new NLLLoss(reduction: Reduction.None);
		private static readonly Scalar one = 1;
		public static Tensor FastCrossEntropyLoss(Tensor input, Tensor logits, double squareboost, bool average, Tensor? boost = null, double gamma = 0.0, bool allow_unsupervised = false) {
			using(NewDisposeScope()){
				Tensor z;
				using(Tensor y = input.exp()){
					z = y.sum(-1, false);
				}
				using(Tensor y = z){
					z = y.log();
				}
				Tensor x;
				if(allow_unsupervised){
					(Tensor v, Tensor i) = input.max(1, true);
					i.Dispose();
					using(v){
						x = cat(new Tensor[] {v, input}, 1);
					}
					using(Tensor y = x){
						x = nlloss.forward(y, logits);
					}
					using (Tensor y = x)
					{
						using(z){
							x = z.add(y);
						}
					}
				} else{
					using Tensor y = nlloss.forward(input, logits);
					using (z)
					{
						x = z.add(y);
					}
				}
				if(gamma > 0.0){
					Tensor x3;
					using(Tensor y = x.negative()){
						x3 = y.exp();
					}
					using(Tensor y = x3){
						x3 = one - y;
					}
					if(gamma != 1.0){
						using Tensor y = x3;
						x3 = y.pow(gamma);
					}
					using(Tensor y = x){
						using(x3){
							x = y.mul(x3);
						}
					}
				}


				if (squareboost > 0)
				{
					using Tensor y = x;
					x = y.addcmul(x, x, squareboost);
				}
				if (boost is { })
				{
					using Tensor y = x;
					x = y.mul(boost);
				}
				if (average)
				{
					using Tensor y = x;
					x = y.mean();
				}
				else
				{
					using Tensor y = x;
					x = y.sum();
				}
				return x.MoveToOuterDisposeScope();


			}
		}

		public static void L2RegularizeIMPL(Tensor? tensor, Scalar lambda)
		{

			if (tensor is null) throw new ArgumentNullException(nameof(tensor));

			(tensor.grad() ?? throw new Exception("No gradients to regularize")).add_(tensor, lambda);

		}
		public static Linear CreateXavierInitializedLinear(int inputs, int outputs, bool bias, double gain = 1.0)
		{
			Linear linear = Linear(inputs, outputs, bias);
			init.xavier_normal_(linear.weight ?? throw new Exception("No weight found (should not reach here)"), gain);
			return linear;
		}
		public static Linear CreateKaimingInitializedLinear(int inputs, int outputs, bool bias, init.FanInOut fanmode, double gain = 1.0)
		{
			//temptest
			//return CreateXavierInitializedLinear(inputs, outputs, bias, gain);
			Linear linear = Linear(inputs, outputs, bias);
			init.kaiming_normal_(linear.weight ?? throw new Exception("No weight found (should not reach here)"), gain, fanmode);
			return linear;
		}
		public static Linear CreateManualKaimingInitializedLinear(int inputs, int outputs, bool bias, int fansize, double gain = 1.0)
		{
			//temptest
			//return CreateXavierInitializedLinear(inputs, outputs, bias, gain);
			Linear linear = Linear(inputs, outputs, bias);
			Tensor w = linear.weight ?? throw new Exception("No weight found (should not reach here)");
			using (no_grad()){
				w.normal_(0.0, gain / Math.Sqrt(fansize));
			}
			return linear;
		}
		public static Linear CreateSparseKaimingInitializedLinear(int inputs, int outputs, bool bias, init.FanInOut fanmode, double gain = 1.0, double dropout = 0.5)
		{
			Linear linear = Linear(inputs, outputs, bias);
			Tensor weight = linear.weight ?? throw new Exception("No weight found (should not reach here)");
			init.kaiming_normal_(weight, gain / (1.0 - dropout), fanmode);
			if(dropout > 0){
				using (no_grad())
				{
					using Dropout dropout1 = Dropout(dropout, true);
					dropout1.forward(weight);
				}
			}
			return linear;
		}
		public static Linear CreateSparseXavierInitializedLinear(int inputs, int outputs, bool bias, double gain = 1.0, double dropout = 0.5)
		{
			Linear linear = Linear(inputs, outputs, bias);
			Tensor weight = linear.weight ?? throw new Exception("No weight found (should not reach here)");
			init.xavier_normal_(weight, gain / (1.0 - dropout));
			if (dropout > 0)
			{
				using (no_grad())
				{
					using Dropout dropout1 = Dropout(dropout, true);
					dropout1.forward(weight);
				}
			}
			return linear;
		}
		public static Conv1d CreateSparseConv(int inputs, int outputs, int kernelSize, double dropout = 0.5)
		{
			Conv1d conv = Conv1d(inputs, outputs, kernelSize);
			if (dropout > 0)
			{
				using (no_grad())
				{
					Tensor weight = conv.weight ?? throw new Exception("Where is my weight? (should not reach here)");
					weight.div_(1.0 - dropout);
					using Dropout dropout1 = Dropout(dropout, true);
					dropout1.forward(weight);

				}
			}
			return conv;
		}
		public static Linear CreateZeroInitializedLinear(int inputs, int outputs, bool bias)
		{
			//temptest
			//return CreateXavierInitializedLinear(inputs, outputs, bias, gain);
			Linear linear = Linear(inputs, outputs, bias);
			Tensor w = (linear.weight ?? throw new Exception("No weight found (should not reach here)"));
			using(no_grad())
			{
				w.zero_();
			}
			return linear;
		}
		public static void EraseReturnAsync<T>(ArrayPool<T> arrayPool, T[] array, int erase) where T : class
		{
			ThreadPool.QueueUserWorkItem(EraseReturn, (arrayPool, array, erase), true);
		}
		private static void EraseReturn<T>((ArrayPool<T> arrayPool, T[] array, int erase) x) where T : class
		{
			for (int i = 0; i < x.erase; ++i)
			{
#pragma warning disable CS8625 // Cannot convert null literal to non-nullable reference type.
				x.array[i] = null;
#pragma warning restore CS8625 // Cannot convert null literal to non-nullable reference type.
			}
			x.arrayPool.Return(x.array);
		}

		public static void AdaptiveLearningRateSGD(Tensor parameter, Tensor momentum, double beta, double baseLearningRate)
		{
			Tensor gradient = parameter.grad() ?? throw new Exception("No grad!");

			using (Tensor abs = momentum.abs())
			{
				abs.mul_(baseLearningRate);
				abs.mul_(gradient);
				parameter.sub_(abs);
			}


			momentum.mul_(beta);
			using (Tensor sign = gradient.sign())
			{
				momentum.add_(sign, 1 - beta);
			}
		}
		public static Tensor MixedPrecisionAttention(Tensor query, Tensor key, Tensor value, Tensor? mask = null, bool causal = false, double dropout = 0.0)
		{
			ScalarType scalarType = query.dtype;
			if (scalarType == ScalarType.Float64 | scalarType == ScalarType.Float32)
			{
				return scaled_dot_product_attention(query, key, value, mask, dropout, causal);
			}
			using (NewDisposeScope())
			{
				Tensor x;
				{
					using Tensor q = query.to(ScalarType.Float32);
					using Tensor k = key.to(ScalarType.Float32);
					using Tensor v = value.to(ScalarType.Float32);
					x = scaled_dot_product_attention(q, k, v, mask, dropout, causal);
				}
				using (x)
				{
					return x.to(scalarType).MoveToOuterDisposeScope();
				}
			}
		}
		public static IEnumerable<Parameter> GradOnly(IEnumerable<Parameter> parameters){
			foreach (Parameter parameter in parameters){
				if (parameter.requires_grad)
					yield return parameter;
			}
		}
		public static Tensor GenerateKaimingQueryMatrix(int inputs, int outputs, int heads, ScalarType? scalarType = null, Device? device = null, bool require_grad = false, double initial_gain = 1.0){
			Span<long> sizes = stackalloc long[4];
			sizes[0] = 1;
			sizes[1] = heads;
			sizes[2] = inputs;
			sizes[3] = outputs;
			return normal(0, initial_gain / Math.Sqrt(inputs), sizes, scalarType, device, require_grad);
		}
		public static Tensor GenerateZeroQueryMatrix(int inputs, int outputs, int heads, ScalarType? scalarType = null, Device? device = null, bool require_grad = false)
		{
			Span<long> sizes = stackalloc long[3];
			sizes[0] = heads;
			sizes[1] = inputs;
			sizes[2] = outputs;
			return zeros(sizes, scalarType, device, require_grad);
		}
		public static IEnumerable<T> JoinEnumerators<T>(IEnumerable<T> enumerator, T append){
			yield return append;
			foreach(T t in enumerator){
				yield return t;
			}
		}
	}
}

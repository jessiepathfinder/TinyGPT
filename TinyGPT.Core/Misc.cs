using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
		public static Dictionary<ushort,double>?[] LoadSimpleDecoder(int tokenclasses, Stream str){
			Dictionary<ushort, double>?[] arr = new Dictionary<ushort, double>[tokenclasses];
			Span<byte> span = stackalloc byte[10];
			ref ushort upf = ref MemoryMarshal.Cast<byte, ushort>(span)[0];
			ref double upd = ref MemoryMarshal.Cast<byte, double>(span.Slice(2, 8))[0];

			
			while(true){
				str.ReadExactly(span);
				ushort ms = upf;
				if (ms == 1) return arr;
				Dictionary<ushort, double> dict = new Dictionary<ushort, double>();
				arr[ms] = dict;
				while(true){
					str.ReadExactly(span);
					ushort ms1 = upf;
					if (ms1 == 0) break;
					dict.Add(ms1, upd);
				}
			}
		}
		public static double[,] SimpleDecodePreLog(Dictionary<ushort, double>?[] spd, ReadOnlySpan<ushort> input)
		{
			int len = input.Length;
			int tcz = spd.Length;
			double[,] doubles = new double[len, tcz];
			for (int i = 0; i < len; ++i)
			{
				Dictionary<ushort, double>? dict = spd[input[i]];
				if (dict is null)
				{
					for (int z = 0; z < tcz;) doubles[i, z++] = 1.0;
				}
				else
				{
					foreach (KeyValuePair<ushort, double> kvp in dict)
					{
						doubles[i, kvp.Key] = kvp.Value;
					}
				}
			}
			return doubles;
		}
		public static float[,] SimpleDecodePreLogFloat(Dictionary<ushort, double>?[] spd, ReadOnlySpan<ushort> input)
		{
			int len = input.Length;
			int tcz = spd.Length;
			float[,] doubles = new float[len, tcz];
			for (int i = 0; i < len; ++i)
			{
				Dictionary<ushort, double>? dict = spd[input[i]];
				if (dict is null)
				{
					for (int z = 0; z < tcz;) doubles[i, z++] = 1.0f;
				}
				else
				{
					foreach (KeyValuePair<ushort, double> kvp in dict)
					{
						doubles[i, kvp.Key] = (float)kvp.Value;
					}
				}
			}
			return doubles;
		}
		public static double[]? TrySimpleDecode(Dictionary<ushort, double>?[] spd, int tokenClasses, ushort prev){
			Dictionary<ushort, double>? dict = spd[prev];
			if (dict is null) return null;
			double[] doubles = new double[tokenClasses];
			foreach (KeyValuePair<ushort, double> kvp in dict)
			{
				doubles[kvp.Key] = (float)kvp.Value;
			}
			return doubles;
		}
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
		public static Tensor FastCrossEntropyLoss(Tensor input, Tensor logits, double squareboost, bool average, Tensor? boost = null, double gamma = 0.0, bool allow_unsupervised = false, bool numfix = false) {
			using(NewDisposeScope()){
				Tensor z = input.logsumexp(-1, false);
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
				if(numfix){
					using Tensor y = x;
					x = torch.nan_to_num(y, 0.0, 0.0, 0.0);
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
		public static Tensor FastSoftmax(Tensor input, Tensor logits)
		{
			using (NewDisposeScope())
			{

				Tensor x;

				
				using (Tensor z = input.logsumexp(-1, false))
				{
					using Tensor y = nlloss.forward(input, logits);
					x = z.add(y);
				}

				using(Tensor y = x){
					x = y.negative();
				}
				using(x){
					return x.exp().MoveToOuterDisposeScope();
				}


			}
		}
		public static void L2RegularizeIMPL(Tensor? tensor, Scalar lambda)
		{

			if (tensor is null) throw new ArgumentNullException(nameof(tensor));

			(tensor.grad() ?? throw new Exception("No gradients to regularize")).add_(tensor, lambda);

		}
		public static void L1RegularizeIMPL(Tensor? tensor, Scalar lambda)
		{

			if (tensor is null) throw new ArgumentNullException(nameof(tensor));
			using Tensor x = tensor.sign();
			(tensor.grad() ?? throw new Exception("No gradients to regularize")).add_(x, lambda);

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
		public static Linear CreatePositiveUniformInitializedLinear(int inputs, int outputs, bool bias, double gain = 1.0)
		{
			//temptest
			//return CreateXavierInitializedLinear(inputs, outputs, bias, gain);
			Linear linear = Linear(inputs, outputs, bias);
			init.uniform_(linear.weight ?? throw new Exception("No weight found (should not reach here)"), 0.0, gain * Math.Sqrt(3.0 / inputs));
			return linear;
		}
		public static Linear CreateManualPositiveUniformInitializedLinear(int inputs, int outputs, bool bias, int fansize, double gain = 1.0)
		{
			//temptest
			//return CreateXavierInitializedLinear(inputs, outputs, bias, gain);
			Linear linear = Linear(inputs, outputs, bias);
			init.uniform_(linear.weight ?? throw new Exception("No weight found (should not reach here)"), 0.0, gain * Math.Sqrt(3.0 / fansize));
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
		public static IEnumerable<Tensor> TensorizeParams(IEnumerable<Parameter> parameters)
		{
			foreach (Parameter parameter in parameters)
			{
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
		public static Tensor GenerateKaimingXATKeyMatrix(int inputs, int outputs, int heads, ScalarType? scalarType = null, Device? device = null, bool require_grad = false, double initial_gain = 1.0)
		{
			Span<long> sizes = stackalloc long[4];
			sizes[0] = heads;
			sizes[1] = 1;
			sizes[2] = inputs;
			sizes[3] = outputs;
			return normal(0, initial_gain / Math.Sqrt(inputs), sizes, scalarType, device, require_grad);
		}
		public static Tensor GenerateKaimingXATValueMatrix(int inputs, int outputs, int heads, ScalarType? scalarType = null, Device? device = null, bool require_grad = false, double initial_gain = 1.0)
		{
			Span<long> sizes = stackalloc long[4];
			sizes[0] = heads;
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
		public static void MakeSecureRandomFloats(Span<float> span){
			int len = span.Length;
			if(len == 0){
				return;
			}
			RandomNumberGenerator.Fill(MemoryMarshal.AsBytes(span));
			Span<uint> uints = MemoryMarshal.Cast<float, uint>(span);
			for(int i = 0; i < len; ++i){
				uints[i] = (uints[i] & 0x3FFFFFFF) | 0x3F800000;
				span[i] -= 1.0f;
			}
		}
	}
}

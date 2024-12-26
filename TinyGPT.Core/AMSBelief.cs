using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace TinyGPT.Core
{
	public sealed class SGDMomentum : IDisposable
	{
		private Memory<Tensor> parameters;
		private readonly Tensor[] state;

		private readonly Scalar bdecay;
		private readonly double bd;
		public SGDMomentum(IEnumerable<Parameter> parameters1, double beta1)
		{
			Tensor[] tensors = Misc.TensorizeParams(parameters1).ToArray();
			bd = beta1;
			parameters = tensors.AsMemory();
			int size = tensors.Length;
			state = new Tensor[size];
			for (int i = 0; i < size; ++i)
			{
				Tensor param = tensors[i];
				state[i] = zeros_like(param, device: CPU).DetachFromDisposeScope();
			}
			//this.decay3 = 1 - beta1;
			bdecay = beta1;

		}
		public SGDMomentum(IReadOnlyDictionary<string,Tensor> rod, double beta1, out ParameterDict momentums)
		{
			bd = beta1;
			bdecay = beta1;
			Span<char> span1 = stackalloc char[1024];
			int spanlen = 1024;
			Queue<Tensor> paramqueue = new Queue<Tensor>();
			Queue<Tensor> statequeue = new Queue<Tensor>();
			momentums = new();
			foreach (KeyValuePair<string,Tensor> kv in rod){
				Tensor t = kv.Value;
				if (!t.requires_grad) continue;
				paramqueue.Enqueue(t);
				Parameter s = Parameter(zeros_like(t, device: CPU),false);
				s.DetachFromDisposeScope();
				statequeue.Enqueue(s);
				string ky = kv.Key;
				int kl = ky.Length;
				int malloc = kl * 2;
				if (span1.Length < spanlen)
				{
					spanlen = malloc * 2;
					span1 = new char[spanlen];
				}
				int ptr = 0;
				for (int i = 0; i < kl; ++i)
				{
					char mychar = ky[i];
					bool isdot = mychar == '.';
					if (mychar == '_' | isdot)
					{
						span1[ptr++] = '_';
					}
					if (isdot) mychar = 'a';

					span1[ptr++] = mychar;
				}
				momentums.Add(new string(span1[..ptr]), s);
			}
			parameters = paramqueue.ToArray();
			state = statequeue.ToArray();
		}
		public void EraseInvalids()
		{
			Memory<Tensor> tensors = parameters;
			Span<Tensor> span = tensors.Span;
			int ctr = 0;
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{
				Tensor tensor = span[i];
				if (tensor.IsInvalid)
				{
					state[i].Dispose();
				}
				else
				{
					int c = ctr++;
					span[c] = tensor;
					state[c] = state[i];
				}
			}
			parameters = tensors.Slice(0, ctr);

		}
		public void zero_grad()
		{
			ReadOnlySpan<Tensor> tensors = parameters.Span;
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{
				tensors[i].grad()?.zero_();
			}
		}

		public void Step(double learningRate, bool final,bool nesterov)
		{
			double mlr = -learningRate;
			Scalar? nesterov_fix;
			if (nesterov)
			{
				double tbd = bd;
				nesterov_fix = mlr * (1.0 - tbd);
				mlr *= tbd;
			}
			else
			{
				nesterov_fix = null;
			}
			Scalar step_size = mlr;
			Scalar bdecay = this.bdecay;


			ReadOnlySpan<Tensor> tensors = parameters.Span;
			using IDisposable disposable = no_grad();
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{

				Tensor exp_avg_ = state[i];
				Tensor param = tensors[i];
				Device device = param.device;
				Tensor grad = (param.grad() ?? throw new Exception("Where is my grad???"));

				using (NewDisposeScope())
				{
					using Tensor exp_avg = exp_avg_.to(device, true);


					exp_avg.mul_(bdecay);
					exp_avg.add_(grad);
					param.add_(exp_avg, step_size);
					if(nesterov_fix is { }){
						param.add_(grad, nesterov_fix);
					}

					if (final)
					{
						exp_avg.Dispose();
						exp_avg_.Dispose();

						continue;
					}

					using (exp_avg)
					{
						exp_avg_.copy_(exp_avg);
					}

				}



			}
			GC.KeepAlive(disposable);
		}

		public void Dispose()
		{
			foreach (Tensor x in state)
			{
				x.Dispose();
			}
		}
	}
	public sealed class AdaState : Module
	{
		public readonly Tensor exp_avg;
		public readonly Tensor exp_avg_sq;

		public AdaState(string name, Tensor exp_avg, Tensor exp_avg_sq) : base(name)
		{
			this.exp_avg = exp_avg;
			this.exp_avg_sq = exp_avg_sq;
			RegisterComponents();
		}
	}
	public sealed class AdaBelief : IDisposable
	{
		private Memory<Tensor> parameters;
		private readonly (Tensor, Tensor)[] state;

		private readonly double beta2;
		//private readonly double decay3;
		private readonly Scalar beta1s;
		private readonly Scalar beta2s;
		private readonly Scalar decay1;
		private readonly Scalar decay2;
		private readonly Scalar eps;
		private readonly Scalar eps2;
		private readonly double bd1;
		private static readonly Scalar one = 1.0;

		public int step;
		public AdaBelief(IEnumerable<Parameter> parameters1, double beta1, double beta2, double epsilon, double epsilon2)
		{
			Tensor[] tensors = Misc.TensorizeParams(parameters1).ToArray();
			parameters = tensors.AsMemory();
			int size = tensors.Length;
			state = new (Tensor, Tensor)[size];
			for (int i = 0; i < size; ++i)
			{
				Tensor param = tensors[i];
				state[i] = (zeros_like(param, device: CPU).DetachFromDisposeScope(), zeros_like(param, device: CPU).DetachFromDisposeScope());
			}
			//this.decay3 = 1 - beta1;
			beta1s = beta1;
			beta2s = beta2;
			decay1 = 1 - beta1;
			bd1 = beta1;
			decay2 = 1 - beta2;
			this.beta2 = beta2;
			eps = epsilon;
			eps2 = epsilon2;
		}
		public AdaBelief(IReadOnlyDictionary<string, Tensor> dict, double beta1, double beta2, double epsilon, double epsilon2, out ModuleDict<AdaState> sdict)
		{
			Queue<Tensor> tensorqueue = new Queue<Tensor>();
			Queue<(Tensor exp_avg, Tensor exp_avg_sq) > statequeue = new Queue<(Tensor exp_avg, Tensor exp_avg_sq)>();
			sdict = new();
			Span<char> span1 = stackalloc char[1024];
			int spanlen = 1024;
			foreach (KeyValuePair<string,Tensor> kvp in dict){
				Tensor val = kvp.Value;
				if (!val.requires_grad) continue;
				tensorqueue.Enqueue(val);
				Tensor a = zeros_like(val, device: CPU).DetachFromDisposeScope();
				Tensor b = zeros_like(val, device: CPU).DetachFromDisposeScope();
				string ky = kvp.Key;
				int kl = ky.Length;
				int malloc = kl * 2;
				if(span1.Length < spanlen){
					spanlen = malloc * 2;
					span1 = new char[spanlen];
				}
				int ptr = 0;
				for (int i = 0; i < kl; ++i) {
					char mychar = ky[i];
					bool isdot = mychar == '.';
					if (mychar == '_' | isdot){
						span1[ptr++] = '_';
					}
					if (isdot) mychar = 'a';

					span1[ptr++] = mychar;
				}


				sdict.Add(new string(span1[..ptr]), new AdaState("", a, b));
				statequeue.Enqueue((a, b));
			}
			Tensor[] tensors = tensorqueue.ToArray();
			parameters = tensors.AsMemory();
			state = statequeue.ToArray();
			
			//this.decay3 = 1 - beta1;
			beta1s = beta1;
			beta2s = beta2;
			decay1 = 1 - beta1;
			bd1 = beta1;
			decay2 = 1 - beta2;
			this.beta2 = beta2;
			eps = epsilon;
			eps2 = epsilon2;
		}
		public void EraseInvalids()
		{
			Memory<Tensor> tensors = parameters;
			Span<Tensor> span = tensors.Span;
			int ctr = 0;
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{
				Tensor tensor = span[i];
				if (tensor.IsInvalid)
				{
					(Tensor a, Tensor b) = state[i];
					a.Dispose();
					b.Dispose();
				}
				else
				{
					int c = ctr++;
					span[c] = tensor;
					state[c] = state[i];
				}
			}
			parameters = tensors.Slice(0, ctr);

		}
		public void zero_grad()
		{
			ReadOnlySpan<Tensor> tensors = parameters.Span;
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{
				tensors[i].grad()?.zero_();
			}
		}

		public void Step(double learningRate, bool final, bool fastAdaBelief, double random_variance, bool nesterov = false, double flra_strength = 0.0)
		{
			//double stepplusplus = ++step;
			double bsc = 1.0 - Math.Pow(beta2, ++step);
			Scalar bias_correction2 = fastAdaBelief ? bsc : Math.Sqrt(bsc);
			//Scalar bias_correction = (1 - Math.Pow(beta1, stepplusplus)) / (1 - beta1);
			double mlr = -learningRate;
			Scalar? lrc = flra_strength > 0.0 ? (bsc * flra_strength) : null;

			
			Scalar? randvar = random_variance > 0.0 ? (-random_variance) : null;
			Scalar? nesterov_fix;
			if(nesterov){
				double tbd = bd1;
				nesterov_fix = mlr * (1.0 - tbd);
				mlr *= tbd;
			} else{
				nesterov_fix = null;
			}
			Scalar step_size = mlr;
			//Scalar ss2 = mlr * decay3;

			Scalar n1 = -1;
			ReadOnlySpan<Tensor> tensors = parameters.Span;
			using IDisposable disposable = no_grad();
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{

				(Tensor exp_avg, Tensor exp_avg_sq) mystate = state[i];
				Tensor param = tensors[i];
				Device device = param.device;
				Tensor grad = (param.grad() ?? throw new Exception("Where is my grad???"));

				using (NewDisposeScope())
				{
					Tensor exp_avg = mystate.exp_avg.to(device, true);
					Tensor exp_avg_sq = mystate.exp_avg_sq.to(device, true);


					exp_avg.mul_(beta1s).add_(grad, alpha: decay1);
					exp_avg_sq.mul_(beta2s);
					//exp_avg_sq.addcmul_(grad, grad, value: decay2);
					using (Tensor x = exp_avg.sub(grad))
					{
						using (Tensor z = x.clone()) x.mul_(z);
						exp_avg_sq.add_(x, decay2);
						if(lrc is { }){
							//Emergency LR Adjustment
							//Allows the Adaptive Learning Rate to be decreased quickly in the event of exploding gradients
							x.mul_(lrc);
							using Tensor y = exp_avg_sq;
							exp_avg_sq = y.maximum(x);
						}

					}


					using (Tensor x = fastAdaBelief ? exp_avg_sq.div(bias_correction2).add_(eps2) : exp_avg_sq.sqrt().div_(bias_correction2).add_(eps))
					{
						if (randvar is { })
						{
							using Tensor x2 = rand_like(x);
							x2.mul_(randvar);
							x2.add_(one);
							x.div_(x2);
						}
						if(nesterov_fix is { }) param.addcdiv_(grad, x, value: nesterov_fix);
						param.addcdiv_(exp_avg, x, value: step_size);
					}


					if (final)
					{
						exp_avg.Dispose();
						exp_avg_sq.Dispose();
						mystate.exp_avg.Dispose();
						mystate.exp_avg_sq.Dispose();
						continue;
					}

					using (exp_avg)
					{
						mystate.exp_avg.copy_(exp_avg);
					}
					using (exp_avg_sq)
					{
						mystate.exp_avg_sq.copy_(exp_avg_sq);
					}
				}



			}
			GC.KeepAlive(disposable);
		}

		public void Dispose()
		{
			foreach ((Tensor x, Tensor y) in state)
			{
				x.Dispose();
				y.Dispose();
			}
		}
	}
	public sealed class AdaMaxSimple : IDisposable
	{
		private Memory<Parameter> parameters;
		private readonly (Tensor, Tensor)[] state;

		//private readonly double decay3;
		private readonly Scalar beta1s;
		private readonly Scalar decay1;
		private readonly double bd1;
		private static readonly Scalar one = 1.0;

		public AdaMaxSimple(IEnumerable<Parameter> parameters1, double beta1, double epsilon)
		{
			Parameter[] tensors = parameters1.ToArray();
			parameters = tensors.AsMemory();
			int size = tensors.Length;
			state = new (Tensor, Tensor)[size];
			Scalar eps = epsilon;
			for (int i = 0; i < size; ++i)
			{
				Parameter param = tensors[i];
				state[i] = (empty_like(param, device: CPU).fill_(eps).DetachFromDisposeScope(), empty_like(param, device: CPU).fill_(eps).DetachFromDisposeScope());
			}
			//this.decay3 = 1 - beta1;
			beta1s = beta1;
			decay1 = 1 - beta1;
			bd1 = beta1;
			
		}
		public void EraseInvalids()
		{
			Memory<Parameter> tensors = parameters;
			Span<Parameter> span = tensors.Span;
			int ctr = 0;
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{
				Parameter tensor = span[i];
				if (tensor.IsInvalid)
				{
					(Tensor a, Tensor b) = state[i];
					a.Dispose();
					b.Dispose();
				}
				else
				{
					int c = ctr++;
					span[c] = tensor;
					state[c] = state[i];
				}
			}
			parameters = tensors.Slice(0, ctr);

		}
		public void zero_grad()
		{
			ReadOnlySpan<Parameter> tensors = parameters.Span;
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{
				tensors[i].grad()?.zero_();
			}
		}

		public void Step(double learningRate, bool final, bool nesterov)
		{

			double mlr = -learningRate;


			Scalar? nesterov_fix;
			if (nesterov)
			{
				double tbd = bd1;
				nesterov_fix = mlr * (1.0 - tbd);
				mlr *= tbd;
			}
			else
			{
				nesterov_fix = null;
			}
			Scalar step_size = mlr;
			//Scalar ss2 = mlr * decay3;

			ReadOnlySpan<Parameter> tensors = parameters.Span;
			using IDisposable disposable = no_grad();
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{

				(Tensor exp_avg, Tensor inverse_lr) mystate = state[i];
				Parameter param = tensors[i];
				Device device = param.device;
				Tensor grad = (param.grad() ?? throw new Exception("Where is my grad???"));

				using (NewDisposeScope())
				{
					Tensor exp_avg = mystate.exp_avg.to(device, true);
					Tensor inverse_lr = mystate.inverse_lr.to(device, true);

					using (Tensor x = inverse_lr, x1 = grad.abs()) inverse_lr = x1.maximum(x);
					exp_avg.mul_(beta1s).add_(grad, alpha: decay1);




					if (nesterov_fix is { }) param.addcdiv_(grad, inverse_lr, value: nesterov_fix);
					param.addcdiv_(exp_avg, inverse_lr, value: step_size);


					if (final)
					{
						exp_avg.Dispose();
						inverse_lr.Dispose();
						mystate.exp_avg.Dispose();
						mystate.inverse_lr.Dispose();
						continue;
					}

					using (exp_avg)
					{
						mystate.exp_avg.copy_(exp_avg);
					}
					using (inverse_lr)
					{
						mystate.inverse_lr.copy_(inverse_lr);
					}
				}



			}
			GC.KeepAlive(disposable);
		}

		public void Dispose()
		{
			foreach ((Tensor x, Tensor y) in state)
			{
				x.Dispose();
				y.Dispose();
			}
		}
	}
	public sealed class Adam : IDisposable
	{
		private Memory<Parameter> parameters;
		private readonly (Tensor, Tensor)[] state;

		private readonly double beta2;
		//private readonly double decay3;
		private readonly Scalar beta1s;
		private readonly Scalar beta2s;
		private readonly Scalar decay1;
		private readonly Scalar decay2;
		private readonly Scalar eps;
		private readonly Scalar eps2;
		private static readonly Scalar one = 1.0;

		private int step;
		public Adam(IEnumerable<Parameter> parameters1, double beta1, double beta2, double epsilon, double epsilon2)
		{
			Parameter[] tensors = parameters1.ToArray();
			parameters = tensors.AsMemory();
			int size = tensors.Length;
			state = new (Tensor, Tensor)[size];
			for (int i = 0; i < size; ++i)
			{
				Parameter param = tensors[i];
				state[i] = (zeros_like(param, device: CPU).DetachFromDisposeScope(), zeros_like(param, device: CPU).DetachFromDisposeScope());
			}
			//this.decay3 = 1 - beta1;
			beta1s = beta1;
			beta2s = beta2;
			decay1 = 1 - beta1;
			decay2 = 1 - beta2;
			this.beta2 = beta2;
			eps = epsilon;
			eps2 = epsilon2;
		}
		public void EraseInvalids()
		{
			Memory<Parameter> tensors = parameters;
			Span<Parameter> span = tensors.Span;
			int ctr = 0;
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{
				Parameter tensor = span[i];
				if (tensor.IsInvalid)
				{
					(Tensor a, Tensor b) = state[i];
					a.Dispose();
					b.Dispose();
				}
				else
				{
					int c = ctr++;
					span[c] = tensor;
					state[c] = state[i];
				}
			}
			parameters = tensors.Slice(0, ctr);

		}
		public void zero_grad()
		{
			ReadOnlySpan<Parameter> tensors = parameters.Span;
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{
				tensors[i].grad()?.zero_();
			}
		}

		public void Step(double learningRate, bool final, bool strongly_convex_mode, double random_variance, double adabound_min_inverse_lr)
		{
			//double stepplusplus = ++step;
			Scalar bias_correction2 = strongly_convex_mode ? (1 - Math.Pow(beta2, ++step)) : Math.Sqrt(1 - Math.Pow(beta2, ++step));
			//Scalar bias_correction = (1 - Math.Pow(beta1, stepplusplus)) / (1 - beta1);
			double mlr = -learningRate;
			Scalar step_size = mlr;
			Scalar? randvar = random_variance > 0.0 ? (-random_variance) : null;
			Scalar? slilr = (adabound_min_inverse_lr > 0.0 & !strongly_convex_mode) ? adabound_min_inverse_lr : null;

			Scalar meps = strongly_convex_mode ? eps2 : eps;

			//Scalar ss2 = mlr * decay3;

			ReadOnlySpan<Parameter> tensors = parameters.Span;
			using IDisposable disposable = no_grad();
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{

				(Tensor exp_avg, Tensor exp_avg_sq) mystate = state[i];
				Parameter param = tensors[i];
				Device device = param.device;
				Tensor grad = (param.grad() ?? throw new Exception("Where is my grad???"));

				using (NewDisposeScope())
				{
					Tensor exp_avg = mystate.exp_avg.to(device, true);
					Tensor exp_avg_sq = mystate.exp_avg_sq.to(device, true);


					exp_avg.mul_(beta1s).add_(grad, alpha: decay1);
					exp_avg_sq.mul_(beta2s);
					exp_avg_sq.addcmul_(grad, grad, value: decay2);


					using (Tensor x = strongly_convex_mode ? exp_avg_sq.div(bias_correction2) : exp_avg_sq.sqrt())
					{
						if (!strongly_convex_mode) x.div_(bias_correction2);
						if (slilr is null)
						{
							x.add_(meps);
						}
						else
						{
							x.clamp_min_(slilr);
						}
						if (randvar is { })
						{
							using Tensor x2 = rand_like(x);
							x2.mul_(randvar);
							x2.add_(one);
							x.div_(x2);
						}
						param.addcdiv_(exp_avg, x, value: step_size);
					}


					if (final)
					{
						exp_avg.Dispose();
						exp_avg_sq.Dispose();
						mystate.exp_avg.Dispose();
						mystate.exp_avg_sq.Dispose();
						continue;
					}

					using (exp_avg)
					{
						mystate.exp_avg.copy_(exp_avg);
					}
					using (exp_avg_sq)
					{
						mystate.exp_avg_sq.copy_(exp_avg_sq);
					}
				}



			}
			GC.KeepAlive(disposable);
		}

		public void Dispose()
		{
			foreach ((Tensor x, Tensor y) in state)
			{
				x.Dispose();
				y.Dispose();
			}
		}
	}
}

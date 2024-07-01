using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace TinyGPT.Core
{

	public sealed class AdaBelief : IDisposable
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
		public AdaBelief(IEnumerable<Parameter> parameters1, double beta1, double beta2, double epsilon, double epsilon2)
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

		public void Step(double learningRate, bool final, bool fastAdaBelief, double random_variance)
		{
			//double stepplusplus = ++step;
			Scalar bias_correction2 = fastAdaBelief ? (1 - Math.Pow(beta2, ++step)) : Math.Sqrt(1 - Math.Pow(beta2, ++step));
			//Scalar bias_correction = (1 - Math.Pow(beta1, stepplusplus)) / (1 - beta1);
			double mlr = -learningRate;
			Scalar step_size = mlr;
			Scalar? randvar = random_variance > 0.0 ? (-random_variance) : null;
			//Scalar ss2 = mlr * decay3;

			Scalar n1 = -1;
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
					//exp_avg_sq.addcmul_(grad, grad, value: decay2);
					using (Tensor x = exp_avg.sub(grad))
					{
						exp_avg_sq.addcmul_(x, x, value: decay2);
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

		public void Step(double learningRate, bool final, bool strongly_convex_mode, double random_variance)
		{
			//double stepplusplus = ++step;
			Scalar bias_correction2 = strongly_convex_mode ? (1 - Math.Pow(beta2, ++step)) : Math.Sqrt(1 - Math.Pow(beta2, ++step));
			//Scalar bias_correction = (1 - Math.Pow(beta1, stepplusplus)) / (1 - beta1);
			double mlr = -learningRate;
			Scalar step_size = mlr;
			Scalar? randvar = random_variance > 0.0 ? (-random_variance) : null;
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


					using (Tensor x = strongly_convex_mode ? exp_avg_sq.div(bias_correction2).add_(eps2) : exp_avg_sq.sqrt().div_(bias_correction2).add_(eps))
					{
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

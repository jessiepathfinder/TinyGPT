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
	public sealed class AdaBelief
	{
		private readonly ReadOnlyMemory<Tensor> parameters;
		private readonly (Tensor,Tensor)[] state;

		private readonly double beta2;
		//private readonly double decay3;
		private readonly Scalar beta1s;
		private readonly Scalar beta2s;
		private readonly Scalar decay1;
		private readonly Scalar decay2;
		private readonly Scalar eps;

		private int step;
		public double learningRate;
		public AdaBelief(IEnumerable<Parameter> parameters1, double beta1, double beta2, double epsilon){
			Tensor[] tensors = parameters1.ToArray();
			parameters = tensors;
			int size = tensors.Length;
			state = new (Tensor, Tensor)[size];
			for (int i = 0; i < size; ++i) {
				Tensor param = tensors[i];
				state[i] = (zeros_like(param, device: CPU), zeros_like(param, device: CPU));
			}
			//this.decay3 = 1 - beta1;
			beta1s = beta1;
			beta2s = beta2;
			decay1 = 1 - beta1;
			decay2 = 1 - beta2;
			this.beta2 = beta2;
			eps = epsilon;
		}
		public void zero_grad(){
			ReadOnlySpan<Tensor> tensors = parameters.Span;
			for (int i = 0, size = tensors.Length; i < size; ++i)
			{
				tensors[i].grad()?.zero_();
			}
		}

		public void Step(){
			//double stepplusplus = ++step;
			Scalar bias_correction2 = Math.Sqrt(1 - Math.Pow(beta2, ++step));
			//Scalar bias_correction = (1 - Math.Pow(beta1, stepplusplus)) / (1 - beta1);
			double mlr = -learningRate;
			Scalar step_size = mlr;
			//Scalar ss2 = mlr * decay3;
			ReadOnlySpan<Tensor> tensors = parameters.Span;
			using IDisposable disposable = no_grad();
			for (int i = 0, size = tensors.Length; i < size; ++i){

				ref (Tensor exp_avg, Tensor exp_avg_sq) mystate = ref state[i];
				Tensor param = tensors[i];
				Device device = param.device;
				Tensor grad = (param.grad() ?? throw new Exception("Where is my grad???"));

				using(NewDisposeScope()){
					Tensor exp_avg;
					using(Tensor x = mystate.exp_avg){
						exp_avg = x.to(device);
					}
					Tensor exp_avg_sq;
					using (Tensor x = mystate.exp_avg_sq)
					{
						exp_avg_sq = x.to(device);
					}


					exp_avg_sq.mul_(beta2s);
					//exp_avg_sq.addcmul_(grad, grad, value: decay2);
					using (Tensor x = exp_avg.sub(grad)){
						exp_avg_sq.addcmul_(x, x, value: decay2);
					}


					exp_avg.mul_(beta1s).add_(grad, alpha: decay1);
					


					
					using(Tensor x = exp_avg_sq.sqrt()){
						x.div_(bias_correction2);
						x.add_(eps);
						//Nesterov correctiom
						//param.addcdiv_(grad, x, value: ss2);
						param.addcdiv_(exp_avg, x, value: step_size);
					}


					using(Tensor x = exp_avg){
						exp_avg = x.to(CPU);
					}
					using (Tensor x = exp_avg_sq)
					{
						exp_avg_sq = x.to(CPU);
					}
					mystate = (exp_avg.DetachFromDisposeScope(), exp_avg_sq.DetachFromDisposeScope());


				}
				

				
			}
			GC.KeepAlive(disposable);
		}

	}
}

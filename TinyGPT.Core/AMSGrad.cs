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
	public sealed class AMSGrad
	{
		private readonly ReadOnlyMemory<Tensor> parameters;
		private readonly (Tensor,Tensor,Tensor)[] state;

		private readonly double beta2;
		private readonly Scalar beta1s;
		private readonly Scalar beta2s;
		private readonly Scalar decay1;
		private readonly Scalar decay2;
		private readonly Scalar eps;

		private int step;
		public double learningRate;
		public AMSGrad(IEnumerable<Parameter> parameters1, double beta1, double beta2, double epsilon){
			Tensor[] tensors = parameters1.ToArray();
			parameters = tensors;
			int size = tensors.Length;
			state = new (Tensor, Tensor, Tensor)[size];
			for (int i = 0; i < size; ++i) {
				Tensor param = tensors[i];
				state[i] = (zeros_like(param, device: CPU), zeros_like(param, device: CPU), zeros_like(param, device: CPU));
			}
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
			Scalar bias_correction2 = Math.Sqrt(1 - Math.Pow(beta2, ++step));
			Scalar step_size = -learningRate;
			ReadOnlySpan<Tensor> tensors = parameters.Span;
			using IDisposable disposable = no_grad();
			for (int i = 0, size = tensors.Length; i < size; ++i){

				ref (Tensor exp_avg, Tensor exp_avg_sq, Tensor max_exp_avg_sq) mystate = ref state[i];
				Tensor param = tensors[i];
				Tensor grad = (param.grad() ?? throw new Exception("Where is my grad???"));

				using(NewDisposeScope()){
					Tensor exp_avg;
					using(Tensor x = mystate.exp_avg){
						exp_avg = x.to(CUDA);
					}
					Tensor exp_avg_sq;
					using (Tensor x = mystate.exp_avg_sq)
					{
						exp_avg_sq = x.to(CUDA);
					}
					Tensor max_exp_avg_sq;
					using (Tensor x = mystate.max_exp_avg_sq)
					{
						max_exp_avg_sq = x.to(CUDA);
					}
					//using Tensor grad = grad2.to(CUDA);

					



					exp_avg.mul_(beta1s).add_(grad, alpha: decay1);
					exp_avg_sq.mul_(beta2s).addcmul_(grad, grad, value: decay2);

					using (Tensor x = max_exp_avg_sq){
						max_exp_avg_sq = maximum(x, exp_avg_sq);
					}
					
					using(Tensor x = max_exp_avg_sq.sqrt()){
						x.div_(bias_correction2);
						x.add_(eps);
						param.addcdiv_(exp_avg, x, value: step_size);

					}

					using(Tensor x = exp_avg){
						exp_avg = x.to(CPU);
					}
					using (Tensor x = exp_avg_sq)
					{
						exp_avg_sq = x.to(CPU);
					}
					using (Tensor x = max_exp_avg_sq)
					{
						max_exp_avg_sq = x.to(CPU);
					}
					mystate = (exp_avg.DetachFromDisposeScope(), exp_avg_sq.DetachFromDisposeScope(), max_exp_avg_sq.DetachFromDisposeScope());


				}
				

				
			}
			GC.KeepAlive(disposable);
		}

	}
}

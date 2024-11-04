using Newtonsoft.Json;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.Serialization.Formatters;
using System.Text;
using System.Transactions;
using TinyGPT.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Transformer = TinyGPT.Core.Transformer;

namespace TinyGPT.Chatbot.NG
{
	internal static class Program
	{
		private sealed class BoosterStage : FullGPTDecoderUnit
		{
			private readonly GPTDecoderUnitV1_1 decoder;
			private readonly Tensor perceptron;
			private static readonly Scalar zero = 0.0;
			public BoosterStage(string name, GPTDecoderUnitV1_1 decoder, Tensor perceptron) : base(name)
			{
				this.decoder = decoder;
				this.perceptron = perceptron;
				RegisterComponents();
			}

			public override Tensor Forward(ReadOnlySpan<ushort> input)
			{
				Tensor[] merge = new Tensor[2];
				using (NewDisposeScope())
				{
					Tensor x;
					using (Tensor y = decoder.Forward(input, input.Length - 1, 0.0, false, true)) x = y.squeeze(0);
					using (Tensor y = x)
					{
						using Tensor y1 = y.clamp_max(zero);
						merge[0] = y1;
						merge[1] = y.relu_();
						x = cat(merge, 0);
					}
					using (x)
					{
						return x.matmul(perceptron).MoveToOuterDisposeScope();
					}

				}
			}
		}
		static void Main(string[] args)
		{
			string? topn_str = Environment.GetEnvironmentVariable("TinyGPT_expand_top_n");

			int expandLimit = topn_str is null ? 16 : Convert.ToInt32(topn_str);
			topn_str = Environment.GetEnvironmentVariable("TinyGPT_open_list_size");

			int listLimit = topn_str is null ? 1024 : Convert.ToInt32(topn_str);
			string datadir = args[0];
			string model = args[1];
			if (!datadir.EndsWith(Path.DirectorySeparatorChar))
			{
				datadir += Path.DirectorySeparatorChar;
			}
			Console.WriteLine("Loading dictionary...");
			IReadOnlyDictionary<string, ushort>? dict = JsonConvert.DeserializeObject<IReadOnlyDictionary<string, ushort>>(File.ReadAllText(datadir + "encoder.json"));
			if (dict is null)
			{
				Console.WriteLine("Null encoder dictionary");
				return;
			}
			int maxlen = 0;
			int tokenclasses = 0;
			foreach (KeyValuePair<string, ushort> keyValuePair in dict)
			{
				maxlen = Math.Max(maxlen, keyValuePair.Key.Length);
				int val = keyValuePair.Value;
				tokenclasses = Math.Max(val, tokenclasses);
			}

			Console.WriteLine("Optimizing dictionary...");
			IReadOnlyDictionary<string, OptimizedTokenizerEntry> optidict = Misc.OptimizeDictionary(dict);

			int magicTokenClasses;
			Console.WriteLine("Loading model...");
			bool usecuda = cuda_is_available();
			if (usecuda)
			{
				InitializeDeviceType(DeviceType.CUDA);
				backends.cuda.matmul.allow_tf32 = true;
				backends.cuda.matmul.allow_fp16_reduced_precision_reduction = false;
				backends.cuda.enable_math_sdp(false);
				backends.cuda.enable_flash_sdp(true);
				backends.cudnn.allow_tf32 = true;

			}
			else
			{
				Console.WriteLine("WARNING: Your computer does not have a NVIDIA CUDA Graphics Card, or lacks the needed graphics drivers (I will be very slow)");
			}
			set_default_dtype(ScalarType.BFloat16);
			FullGPTDecoderUnit themodel;
			int maxcontext;
			switch (model)
			{
				case "nano-v2":
				case "nano-v1_1":
					{
						const int latentTokenSize = 2048;
						maxcontext = 1025;
						const int attentionHeads = 16;
						const int firstTierAttentionDepth = 5;
						magicTokenClasses = 4;
						tokenclasses += magicTokenClasses + 1;
						themodel = new GPTDecoderUnitV1_1("TinyGPT", latentTokenSize, attentionHeads, firstTierAttentionDepth, 0.0, 1e-6, 1024, 0.0, 1.0, 1.0, 0.0, tokenclasses, 1.0, 128, 0.0, 1, 2048, 0.0, 4);
					}
					break;
				case "nano-v1":

					{
						const int latentTokenSize = 2048;
						maxcontext = 1025;
						const int attentionHeads = 16;
						const int firstTierAttentionDepth = 5;
						magicTokenClasses = 4;
						tokenclasses += magicTokenClasses + 1;
						themodel = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, firstTierAttentionDepth, 0.0, 1e-6, 1024, 0.0, 1.0, 1.0, 0.0, tokenclasses, 1.0, 128, 0.0, 1, 2048, 0.0);

					}
					break;
				case "nano-v1_2":

					{
						const int latentTokenSize = 2048;
						maxcontext = 1025;
						const int attentionHeads = 16;
						const int firstTierAttentionDepth = 5;
						magicTokenClasses = 3;
						tokenclasses += magicTokenClasses + 1;
						themodel = new GPTDecoderUnitV1_2("TinyGPT", latentTokenSize, attentionHeads, firstTierAttentionDepth, 1e-7, 1.0, 1.0, 0.0, tokenclasses, 1.0, 128, 0.0, 2, 2048, 0.0, 2, empty(tokenclasses, latentTokenSize), false, false, 0);

					}
					break;
				default:
					Console.WriteLine("unknown model!");
					return;
			}
			themodel.to(ScalarType.BFloat16);
			themodel.load(datadir + model + ".model");
			if (themodel is GPTDecoderUnitV1_1 ncg && Environment.GetEnvironmentVariable("TinyGPT_booster_enabled") == "1")
			{
				themodel = new BoosterStage("", ncg, Tensor.load(datadir + model + ".booster.model"));
			}
			foreach (Parameter parameter in themodel.parameters())
			{
				parameter.requires_grad = false;
			}
			themodel.eval();

			if (usecuda)
			{
				themodel.to(CUDA);
			}
			//4 magic token types

			string[] decode = new string[tokenclasses + 1];
			int maxtokensize = 0;
			foreach (KeyValuePair<string, ushort> keyValuePair in dict)
			{
				string key = keyValuePair.Key;
				decode[keyValuePair.Value + magicTokenClasses] = key;
				maxtokensize = Math.Max(maxtokensize, key.Length);
			}

			Span<ushort> buffer = stackalloc ushort[maxcontext];

			int maxinsize = maxcontext - 2;


			Console.WriteLine("Ready!");
			while (true)
			{
				Console.Write("User: ");
				string? input = Console.ReadLine();
				if (input is null)
				{
					continue;
				}

				Console.Write("TinyGPT: ");
				int intokens = Transformer.Tokenize(optidict, buffer, input, maxtokensize, magicTokenClasses);
				if (intokens > maxinsize)
				{
					Console.WriteLine("too big!");
					continue;
				}
				int it1 = intokens + 1;

				
				
				State?[] openList = new State?[listLimit];
				openList[0] = new State(buffer.Slice(0, intokens).ToArray(), 0, 0.0);
				int openListCircularAssist = 0;
				Queue<(ReadOnlyMemory<ushort> memory, double loss)> solutions = new Queue<(ReadOnlyMemory<ushort> memory, double loss)>();
				while(true){
					double lloss = double.PositiveInfinity;
					int delete = -1;
					for(int i = 0; i < listLimit; ++i){
						State? s1 = openList[i];
						if (s1 is null) continue;
						double myloss = s1.total_loss / ((s1.state.Length - intokens) + 1);
						if(myloss < lloss){
							lloss = myloss;
							delete = i;
						}
					}
					if (delete < 0) break;
					State s = openList[delete] ?? throw new Exception("Unexpected null state selected (should not reach here)");
					ReadOnlyMemory<ushort> myprev = s.state;
					
					int mylen = myprev.Length;
					//Console.WriteLine(mylen);

					bool isCriticalLen = mylen == maxcontext;
					openList[delete] = null;

					double myloss1 = s.total_loss;
					int oml = mylen;
					ushort[] commit = new ushort[++mylen];
					myprev.CopyTo(commit);
					commit[oml] = s.action;
					double divide1 = (mylen - intokens) + 1;
					Tensor losses;
					Tensor indices;
					ReadOnlyMemory<ushort> crom = commit;
					using (Tensor x = themodel.Forward(crom.Span))
					{
						(losses, indices) = x.sort(0, true, false);
					}
					using (Tensor x = losses) losses = x.log_softmax(0, ScalarType.Float64);
					if (usecuda)
					{
						using (Tensor x = losses) losses = x.cpu();
						using (Tensor x = indices) indices = x.cpu();
					}
					for (int i = 0; i < expandLimit; ++i)
					{
						double myloss;
						using (Tensor temp1 = losses[i]) myloss = temp1.ToDouble();
						myloss *= -1;
						myloss += myloss1;
						double mldiv = myloss / divide1;
						int myind;
						
						using (Tensor temp1 = indices[i]) myind = temp1.ToInt32();

						string? detokenized = decode[myind];
						ushort ms = (ushort)myind;
						if ((!(detokenized is { } && detokenized.Contains(','))) && commit.AsSpan(Math.Max(it1, mylen - 4)).Contains(ms)) {
							continue;
						}

						bool finished = myind == 1;
						if (finished | isCriticalLen){
							ReadOnlyMemory<ushort> u = myprev[(intokens + 1)..];
							if(!finished){
								int temlen = u.Length;
								ushort[] ushorts = new ushort[temlen + 1];
								u.CopyTo(ushorts);
								ushorts[temlen] = (ushort)myind;
								u = ushorts;
							}
							solutions.Enqueue((u, myloss));
							continue;
						}

						double highestLoss = 0.0;
						int replaceIndex = -1;
						for(int z = 0; z < listLimit; ++z){
							++openListCircularAssist;
							openListCircularAssist %= listLimit;
							State? s1 = openList[openListCircularAssist];
							if(s1 is null){
								replaceIndex = openListCircularAssist;
								break;
							}
							int s1sl = s1.state.Length;
							if (s1sl >= mylen) continue;
							
							double myloss2 = (s1.total_loss) / ((s1sl - intokens) + 1);

							if (myloss2 < mldiv) continue;
							if (myloss2 > highestLoss){
								highestLoss = myloss2;
								replaceIndex = openListCircularAssist;
							}
						}
						if(replaceIndex > -1){
							openList[replaceIndex] = new State(crom, ms, myloss);
						}
					}
					losses.Dispose();
					indices.Dispose();


				}
				if (solutions.Count == 0) throw new Exception("Solution list empty (should not reach here)");
				ReadOnlyMemory<ushort> rom1 = ReadOnlyMemory<ushort>.Empty;
				double bestloss3 = double.PositiveInfinity;
				while(solutions.TryDequeue(out (ReadOnlyMemory<ushort> mem, double loss) x)){
					double myloss4 = x.loss / x.mem.Length;
					if (myloss4 < bestloss3){
						bestloss3 = x.loss;
						rom1 = x.mem;
					}
				}

				ReadOnlySpan<ushort> findump = rom1.Span;
				for (int i = 0, stop = findump.Length; i < stop; ++i){
					int bestindex = findump[i];
					string? str = decode[bestindex];

					str ??= " invalid_word_" + bestindex;
					Console.Write(str);
				}




				Console.WriteLine();
				Console.WriteLine("==================================================");
				Console.WriteLine();

			}
		}
		private sealed record State{
			public readonly double total_loss;
			public readonly ushort action;
			public readonly ReadOnlyMemory<ushort> state;

			public State(ReadOnlyMemory<ushort> state, ushort action, double total_loss)
			{
				this.total_loss = total_loss;
				this.action = action;
				this.state = state;
			}

		}

	}
}
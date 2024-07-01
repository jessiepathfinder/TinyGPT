using Newtonsoft.Json;
using System.Collections;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using Tensorboard;
using TinyGPT.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

using Transformer = TinyGPT.Core.Transformer;

namespace TinyGPT.DecoderV2.Trainer
{
	internal static class Program
	{
		//hyperparameters
		private const int latentTokenSize = 2048;
		private const int maxContextSize = 1025;
		private const int trainingBatches = 100000;
		private const int targetTokensPerBatch = 4096;
		private const int attentionHeads = 16;
		private const int firstTierAttentionDepth = 5;
		private const int magicTokenClasses = 4;
		private const int maxQuestionSize = 256;
		private const double futureRewardDecay = 0.99;
		private const double firstOccouranceBonus = 2.0;
		private const int antiRepetitionLookback = 4;
		private const double deepfakeDetectorWeight = 0.1;
		private const double cumulativeFutureRewardScale = 1.0 - futureRewardDecay;
		private const int episodeSize = 1024;
		private const double temperature = 1.0;
		private const double deepfakeDetectorStepIn = 0.99;
		private const int multiprop = 4;


		private const int mcsm1 = maxContextSize - 1;


		private static GPTDecoderUnitV1_1 CreateDecoder(int tokenclasses, double dropouts)
		{
			return new GPTDecoderUnitV1_1("TinyGPT", latentTokenSize, attentionHeads, firstTierAttentionDepth, 0.0, 1e-7, 1024, 0.0, 1.0, 1.0, dropouts, tokenclasses, 1.0, 128, dropouts, 1, 2048, dropouts,4);
		}

		private static void Main(string[] args)
		{
			string datadir = args[0];
			string save = args[1];
			string pretrained = args[2];
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
				tokenclasses = Math.Max(keyValuePair.Value, tokenclasses);
			}
			//5 magic token types
			//[START_GPT], [END_GPT], [WIKI_SEPERATOR], [MASK]
			tokenclasses += magicTokenClasses + 1;
			int tokenClasses2 = tokenclasses;
			string?[] detokenizer = new string?[tokenclasses];
			foreach (KeyValuePair<string, ushort> keyValuePair in dict)
			{
				maxlen = Math.Max(maxlen, keyValuePair.Key.Length);
				detokenizer[keyValuePair.Value + magicTokenClasses] = keyValuePair.Key;
			}
			detokenizer[0] = "\nTinyGPT: ";

			Console.WriteLine("Optimizing dictionary...");
			IReadOnlyDictionary<string, OptimizedTokenizerEntry>? dict1 = Misc.OptimizeDictionary(dict);



			Console.WriteLine("Loading question bank...");
			Queue<ushort[]>? dataqueue = new Queue<ushort[]>();
			Queue<object>? jit_tokenizer_queue = new Queue<object>();
			//File.ReadAllText(datadir + "QuestionAnsweringV2.jsonl.deflate")

			Span<ushort> span = stackalloc ushort[maxContextSize + 1];
			using (StreamReader reader = new StreamReader(new DeflateStream(new FileStream(datadir + "QuestionAnsweringV2.jsonl.deflate", FileMode.Open, FileAccess.Read, FileShare.Read, 16777216, FileOptions.SequentialScan), CompressionMode.Decompress, false), Encoding.UTF8, false, 16777216, false))
			{
			read:
				string? line = reader.ReadLine();
				if (line is { })
				{

					if(line.StartsWith('!')){
						line = line.Substring(1);
						goto noread;
					}

					string[] myarr = (JsonConvert.DeserializeObject<string[]>(line) ?? throw new Exception("Null array (should not reach here)"));
					string str = myarr[0];
					int c = Transformer.Tokenize(dict1, span, str.AsSpan(), maxlen, magicTokenClasses);
					if(c == 0 | c > maxQuestionSize){
						goto noread;
					}
					dataqueue.Enqueue(span.Slice(0, c).ToArray());


				noread:
					jit_tokenizer_queue.Enqueue(line);
					goto read;
				}
			}
			Console.WriteLine("Optimizing memory usage...");
			ushort[][] questionBank = dataqueue.ToArray();
			int datacount = questionBank.Length;
			object?[] jit_tokenizer_data = jit_tokenizer_queue.ToArray();
			jit_tokenizer_queue = null;
			int jitTokenizerQueueSize = jit_tokenizer_data.Length;
			dataqueue = null;
			GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, true, true);

			Console.WriteLine("Initializing GPU...");
			InitializeDeviceType(DeviceType.CUDA);
			backends.cuda.matmul.allow_tf32 = true;
			backends.cuda.matmul.allow_fp16_reduced_precision_reduction = false;
			backends.cuda.enable_math_sdp(false);
			backends.cuda.enable_flash_sdp(true);
			backends.cudnn.allow_tf32 = true;
			set_default_dtype(ScalarType.BFloat16);

			Console.WriteLine("Initializing student model...");
			GPTDecoderUnitV1_1 student1 = CreateDecoder(tokenclasses, 0.125);
			student1.to(bfloat16);
			student1.load(pretrained);
			student1.to(CUDA);

			Console.WriteLine("Initializing student optimizer...");
			IEnumerable<Parameter> parameters = student1.parameters();
			AdaBelief adaBelief = new AdaBelief(parameters, 0.9, 0.999, 1e-9, 1e-15);


			Console.WriteLine("Initializing teacher model...");
			GPTDecoderUnitV1_1 teacher = CreateDecoder(tokenclasses, 0.0);
			foreach(Parameter parameter in teacher.parameters()){
				parameter.requires_grad = false;
			}
			teacher.to(bfloat16);
			teacher.load(pretrained);
			

			Console.WriteLine("Initializing deepfake detector model and optimizer...");
			double sqrt2 = Math.Sqrt(2.0);
			ModuleList<Module<Tensor, Tensor>> deepfake_detector = new ModuleList<Module<Tensor, Tensor>>()
			{
				Misc.CreateKaimingInitializedLinear(2048, 2048, true, init.FanInOut.FanIn),
				Softplus(),
				Dropout(0.25),
				Misc.CreateKaimingInitializedLinear(2048, 2048, true, init.FanInOut.FanIn),
				Softplus(),
				Dropout(0.25),
				Misc.CreateKaimingInitializedLinear(2048, 2048, true, init.FanInOut.FanIn),
				Softplus(),
				Dropout(0.25),
				Misc.CreateKaimingInitializedLinear(2048, 2048, true, init.FanInOut.FanIn),
				Softplus(),
				Dropout(0.25),
				Misc.CreateKaimingInitializedLinear(2048, 1, true, init.FanInOut.FanIn)
			};
			DoesNothingModule doesNothingModule = new DoesNothingModule("", deepfake_detector);
			doesNothingModule.to(CUDA, bfloat16);
			doesNothingModule.train();
			IEnumerable<Parameter> parameters1 = doesNothingModule.parameters();

			
			AdaBelief deepfake_adaBelief = new AdaBelief(parameters1, 0.9, 0.999, 1e-9, 1e-15);

			/*
			Console.WriteLine("Initializing future rewards predictor...");
			ModuleList<Module<Tensor, Tensor>> rewards_predictor = new ModuleList<Module<Tensor, Tensor>>()
			{
				Misc.CreateKaimingInitializedLinear(2048, 2048, true, init.FanInOut.FanIn,sqrt2),
				ReLU(),
				Dropout(0.25),
				Misc.CreateKaimingInitializedLinear(2048, 2048, true, init.FanInOut.FanIn,sqrt2),
				ReLU(),
				Dropout(0.25),
				Misc.CreateKaimingInitializedLinear(2048, 2048, true, init.FanInOut.FanIn,sqrt2),
				ReLU(),
				Dropout(0.25)
			};

			DoesNothingModule doesNothingModule1 = new DoesNothingModule("", rewards_predictor);
			doesNothingModule1.to(CUDA, bfloat16);
			doesNothingModule1.train();
			*/
			




			Console.WriteLine("Start training!");
			Queue<(ushort[] state, Tensor actions)> replayBuffer = new Queue<(ushort[] state, Tensor actions)>();
			Span<float> tkbuffer = stackalloc float[1024];
			Span<double> rewardbuffer = stackalloc double[maxContextSize];
			Span<ushort> rbf = stackalloc ushort[antiRepetitionLookback];
			int tkwindow = 0;
			Misc.MakeSecureRandomFloats(tkbuffer);
			Queue<string> savequeue = new Queue<string>();
			string note = "Generated {0}/" + targetTokensPerBatch + " tokens";
			Scalar ne1 = -1.0;
			//Scalar deepfakeDetectorL2Regularization = 1e-5;
			Scalar studentWordEmbeddingsDecoupledWeightDecay = 0.999;
			Scalar cfrsa = cumulativeFutureRewardScale;
			long[] reddims = new long[] {1};
			double dfds = 1.0;
			for (int z = 0; z < trainingBatches; ++z){
				Console.WriteLine("Start training batch #{0}...", z);



				Console.WriteLine("Populating replay buffer...");
				set_grad_enabled(false);
				bool starting = true;
				
				int tokensGenerated = 0;
				int deepfakeTokens = 0;
				while (tokensGenerated < targetTokensPerBatch){
					for (int i = 0; i < antiRepetitionLookback;)
					{
						rbf[i++] = 1;
					}
					ushort[] question = questionBank[RandomNumberGenerator.GetInt32(0, datacount)];
					int index = question.Length;
					question.CopyTo(span);
					span[index] = 0; //[STARTGPT]
					int slice = ++index;
					
					bool firsttkn = true;
					double endprob = 1.0;
					while (index < maxContextSize)
					{

						Tensor tensor;
						using (Tensor x = student1.Forward(span[..index]))
						{
							tensor = x.to(float64);
						}
						using (Tensor x = tensor)
						{
							tensor = x.softmax(0);
						}
						if (firsttkn)
						{
							Scalar s;
							using (Tensor x = tensor[1])
								s = x.ToScalar();
							tensor[1] = 0;
							endprob = 1.0 - s.ToDouble();
							firsttkn = false;
						}

						Tensor indices;
						using (Tensor x = tensor)
						{
							(tensor, indices) = x.sort(descending: true);
						}
						using (Tensor x = tensor)
						{
							tensor = x.cpu();
						}
						using (Tensor x = indices)
						{
							indices = x.cpu();
						}
						ushort bestindex;
						using (tensor)
						{
							using (indices)
							{
							tkrestart:

								double tk = tkbuffer[tkwindow] * temperature * endprob;
								if (tkwindow == 1023)
								{
									tkwindow = 0;
									Misc.MakeSecureRandomFloats(tkbuffer);
								}
								else
								{
									++tkwindow;
								}
								int j = 0;
								while (true)
								{

									Scalar mytkn2;
									using (Tensor tt = tensor[j])
										mytkn2 = tt.ToScalar();
									tk -= mytkn2.ToDouble();
									if (tk <= 0.0)
									{
										using (Tensor tt = indices[j])
											mytkn2 = tt.ToScalar();
										bestindex = (ushort)mytkn2.ToInt32();
										break;
									}

									if (++j == tokenclasses)
									{
										goto tkrestart;
									}

								}
							}
						}
						endprob = 1.0;
						span[index++] = bestindex;
						++tokensGenerated;
						if (tokensGenerated % 256 == 0)
						{
							Console.WriteLine(note, tokensGenerated);
						}
						if (bestindex == 1)
						{
							break;
						}

						if (rbf.Contains(bestindex))
						{
							string? detokenized = detokenizer[bestindex];
							if (detokenized is null)
								break;
							if (!detokenized.Contains(','))
								break;
						}
						rbf[index % antiRepetitionLookback] = bestindex;




					}
					if (starting)
					{
						starting = false;
						Console.Write("Autouser: ");
						for (int i = 0; i < index; ++i)
						{
							int bestindex = span[i];
							if (bestindex == 1)
							{
								break;
							}
							string? str = detokenizer[bestindex];

							if (str is null)
							{
								str = " invalid_word_" + bestindex;
							}
							else
							{
								Console.Write(str);
							}
						}
						Console.WriteLine();
						Console.WriteLine("==================================================");
						Console.WriteLine();
					}
					if (index == slice)
					{
						throw new Exception("Index equals slice (should not reach here)");
					}
					deepfakeTokens += index - 1;
					long[] actions = new long[index - slice];
					for (int i = slice; i < index; ++i)
					{
						actions[i - slice] = span[i];
					}
					replayBuffer.Enqueue((span.Slice(0, index).ToArray(), tensor(actions, ScalarType.Int64, CUDA)));
				}
				Console.WriteLine("Migrating teacher to GPU...");
				teacher.to(CUDA);
				double totalDeepfakeDetectorLoss = 0.0;
				

				set_grad_enabled(true);

				int deepfakeTokens1 = 0;
				Console.WriteLine("Data pass deepfake detector...");
				while (deepfakeTokens1 < targetTokensPerBatch)
				{
				start1:
					ref object? obj = ref jit_tokenizer_data[RandomNumberGenerator.GetInt32(0, jitTokenizerQueueSize)];
					ushort[]? arr = obj as ushort[];
					int slice;
					if (arr is null)
					{
						string? str = obj as string;
						if (str is null)
						{
							goto start1;
						}
						string[] myarr = (JsonConvert.DeserializeObject<string[]>(str) ?? throw new Exception("Null array (should not reach here)"));
						slice = Transformer.Tokenize(dict1, span.Slice(1, mcsm1), myarr[0].AsSpan(), maxlen, magicTokenClasses);
						if (slice == 0)
						{
							goto invalid;
						}
						if (slice == maxContextSize)
						{
							goto invalid;
						}
						span[++slice] = 0; //[START_GPT]
						slice += 1;
						int c1 = Transformer.Tokenize(dict1, span.Slice(slice, mcsm1 - slice), myarr[1].AsSpan(), maxlen, magicTokenClasses);
						if (c1 == 0)
						{
							goto invalid;
						}
						span[0] = (ushort)slice;
						arr = span.Slice(0, slice + c1).ToArray();
						obj = arr;
						goto valid;
					invalid:;
						obj = null;
						goto start1;
					valid:;
					} else{
						slice = arr[0];
					}
					int len = arr.Length - 1;
					deepfakeTokens1 += len - slice;
					Tensor x = teacher.Forward(arr.AsSpan(1, len), slice, 0.0, true);
					set_grad_enabled(true);
					foreach (Module<Tensor, Tensor> module in deepfake_detector)
					{
						using Tensor y = x;
						x = module.forward(y);
					}
					using (Tensor y = x) x = y.negative();
					using (Tensor y = x) x = y.softplus();
					
					using (Tensor y = x) x = y.sum();
					using(x){
						x.backward();
						totalDeepfakeDetectorLoss += x.ToScalar().ToDouble();
					}
					set_grad_enabled(false);
				}
				deepfakeTokens += deepfakeTokens1;
				
				Console.WriteLine("Computing rewards...");
				int rbcount = replayBuffer.Count;
				int rbctr = 0;
				(ushort[] state, Tensor actions, Tensor rewards)[] SAR = new (ushort[] state, Tensor actions, Tensor rewards)[rbcount];
				double totalRewards = 0.0;
				while (replayBuffer.TryDequeue(out (ushort[] state, Tensor actions) run)){
					int statelen = run.state.Length;
					
					int actionlen = (int)run.actions.size(0);
					int slice1 = statelen - actionlen;
					int slice = slice1 - 1;

					Tensor y = teacher.Forward(run.state.AsSpan(0, statelen - 1), slice, 0.0, true);
					
					Tensor dfd = y.slice(0, 1, actionlen, 1);
					Tensor deepfakeRewards = dfd;
					Tensor? disposeme = null;
					doesNothingModule.eval();
					foreach (Module<Tensor, Tensor> module in deepfake_detector)
					{
						deepfakeRewards = module.forward(deepfakeRewards);
						disposeme?.Dispose();
						disposeme = deepfakeRewards;
					}
					doesNothingModule.train();
					//using (Tensor x = deepfakeRewards) deepfakeRewards = x.to(float64);
					deepfakeRewards.sigmoid_();
					/*
					double totalDeepfakeRewards;
					using (Tensor x = deepfakeRewards.mean())
						totalDeepfakeRewards = x.ToScalar().ToDouble();
					*/
					using (Tensor x = deepfakeRewards)
						deepfakeRewards = x.cpu();


					set_grad_enabled(true);
					foreach (Module<Tensor, Tensor> module in deepfake_detector)
					{
						using Tensor x = dfd;
						dfd = module.forward(x);
					}
					using (Tensor x = dfd) dfd = x.softplus();
					using (Tensor x = dfd) dfd = x.sum();
					using (dfd)
					{
						dfd.backward();
						totalDeepfakeDetectorLoss += dfd.ToScalar().ToDouble();
					}
					set_grad_enabled(false);
					


					teacher.FinalForward(ref y);
					//y = x.to(float32);
					using (Tensor x = y) y = Misc.FastSoftmax(x, run.actions);
					using (Tensor x = y) y = x.cpu();
					Dictionary<ushort, bool> keyValuePairs = new Dictionary<ushort, bool>();
					double dfr = 1.0;
					double dfm = deepfakeDetectorWeight * (1.0 - dfds);
					double mdfr = dfm;

					
					using (y){
						using(deepfakeRewards){
							for (int i = 0; true;)
							{

								
								double reward;
								using (Tensor x = y[i]) reward = x.ToScalar().ToDouble();

								ushort mytkn = run.state[i + slice1];


								int oi = i;
								bool stopping = ++i == actionlen;
								if (stopping)
								{

									if (mytkn == 1)
									{
										string? detokenize = detokenizer[run.state[oi + slice]];
										if (detokenize is null || !(detokenize.EndsWith('.')))
										{
											reward = 0.0;
											goto norewards;
										}
										reward += mdfr;
										reward /= Math.Pow(futureRewardDecay, oi) + cumulativeFutureRewardScale;
										

										//reward += earlyStoppingBonus * (maxContextSize - statelen);
										goto fastrewards;
									}
									reward = 0.0;
									goto norewards;
								}

								if (keyValuePairs.TryAdd(mytkn, false))
								{
									reward *= firstOccouranceBonus;
								}
								Scalar sdeepfakeReward;
								using (Tensor x = deepfakeRewards[oi])
									sdeepfakeReward = x.ToScalar();
								double myDeepfakeRewards = sdeepfakeReward.ToDouble();
								if(myDeepfakeRewards < dfr){
									dfr = myDeepfakeRewards;
									mdfr = myDeepfakeRewards * dfm;
								}

								reward += mdfr;

							fastrewards:
								totalRewards += reward;
							norewards:
								rewardbuffer[oi] = reward;
								if(stopping){
									break;
								}
							}
						}
					}
					float[] rewards = new float[actionlen];
					//Compute discounted future rewards
					double tr2 = 0.0;
					for(int i = actionlen - 1; i >= 0; --i){
						tr2 *= futureRewardDecay;
						tr2 += rewardbuffer[i];
						rewards[i] = (float)tr2;
					}
					SAR[rbctr++] = (run.state, run.actions, tensor(rewards, ScalarType.Float32, CUDA));
				}
				dfds *= deepfakeDetectorStepIn;
				Console.WriteLine("Migrating teacher to CPU...");
				teacher.to(CPU);
				double deepfakeDetectorLoss = totalDeepfakeDetectorLoss / deepfakeTokens;
				Console.WriteLine("Average deepfake detector loss: {0}", deepfakeDetectorLoss);
				Scalar deepfakeLossDiv = deepfakeTokens;
				Console.WriteLine("Average reward: {0}", totalRewards / rbcount);
				set_grad_enabled(true);
				
				Console.WriteLine("Scaling deepfake detector gradients...");
				foreach(Parameter parameter in parameters1){
					(parameter.grad() ?? throw new Exception("Where is my grad? (should not reach here)")).div_(deepfakeLossDiv);
				}
				Console.WriteLine("Deepfake detector optimizer step...");
				deepfake_adaBelief.Step(1e-4, false, false, 0.0);
				Console.WriteLine("Disposing deepfake detector gradients...");
				foreach (Parameter parameter in parameters1)
				{
					(parameter.grad() ?? throw new Exception("Where is my grad? (should not reach here)")).Dispose();
#pragma warning disable CS8625 // Cannot convert null literal to non-nullable reference type.
					parameter.set_grad(null);
#pragma warning restore CS8625 // Cannot convert null literal to non-nullable reference type.
				}

				
				Console.WriteLine("Computing and backpropagating loss...");
				double totalLosses = 0.0;
				for(int i = 0, stop = rbctr * multiprop; i < stop; ){
					(ushort[] state, Tensor actions, Tensor rewards) = SAR[i++ % rbctr];
					//rewards.mul_(cfrsa);
					int statelen = state.Length - 1;

					/*
					Tensor y = student.Forward(state.AsSpan(0, statelen), (int)(statelen - actions.size(0)), 0.0, true);
					Tensor rewardPotentialPredict = y;
					Tensor? disposeme = null;
					foreach(Module<Tensor, Tensor> module in rewards_predictor){
						rewardPotentialPredict = module.forward(rewardPotentialPredict);
						disposeme?.Dispose();
						disposeme = rewardPotentialPredict;
					}
					using (Tensor x = rewardPotentialPredict) rewardPotentialPredict = x.mean(reddims, false, float32);
					student.FinalForward(ref y);
					using(Tensor x = y)
					{
						y = x.to(ScalarType.Float32);
					}
					using(actions){
						using Tensor x = y;
						y = Misc.FastSoftmax(x, actions);
					}
					using(rewards){
						using(rewardPotentialPredict){
							using Tensor x = y;
							y = rewards.addcmul(rewardPotentialPredict, x, ne1);
						}
					}
					using (Tensor x = y) y = x.mul(x);
					
					using (Tensor x = y) y = x.sum();
					*/

					Tensor y;
					using(Tensor x = student1.Forward(state.AsSpan(0, statelen), (int)(statelen - actions.size(0)), 0.0, false)){
						y = x.to(float32);
					}
					using (Tensor x = y) y = Misc.FastCrossEntropyLoss(x, actions, 0.0, false, rewards);
					using (y) {
						y.backward();
						totalLosses += y.ToScalar().ToDouble();
					}
				}
				for(int i = 0; i < rbctr; ){
					(ushort[] state, Tensor actions, Tensor rewards) = SAR[i++];
					actions.Dispose();
					rewards.Dispose();
				}
				tokensGenerated *= multiprop;
				Console.WriteLine("Average loss: {0}", totalLosses / tokensGenerated);


				/*
				using (no_grad())
				{
					foreach (KeyValuePair<string, Tensor> kvp in student.state_dict())
					{
						Tensor? t = kvp.Value.grad();
						if (t is null)
						{
							continue;
						}
						Tensor mean;
						Tensor std;
						Tensor min;
						Tensor max;
						using (Tensor t2 = t.to(ScalarType.Float32))
						{
							(std, mean) = t2.std_mean(false);
							min = t2.min();
							max = t2.max();
						}

						Console.WriteLine("{0}: std: {1} mean: {2} min: {3} max: {4}", kvp.Key.PadRight(25), std.ToSingle().ToString().PadRight(15), mean.ToSingle().ToString().PadRight(15), min.ToSingle().ToString().PadRight(15), max.ToSingle());
						std.Dispose();
						mean.Dispose();
						min.Dispose();
						max.Dispose();
					}
				}
				*/

				Console.WriteLine("Scaling gradients...");
				Scalar costdiv = tokensGenerated;
				foreach (Parameter parameter in parameters)
				{
					(parameter.grad() ?? throw new Exception("Where is my grad? (should not reach here)")).div_(costdiv);
				}



				Console.WriteLine("Optimizer step...");
				
				adaBelief.Step(1e-5, false, false, 0.0);
				

				Console.WriteLine("Disposing gradients...");
				foreach (Parameter parameter in parameters)
				{
					Tensor grad = parameter.grad() ?? throw new Exception("Where is my grad? (should not reach here)");
					grad.Dispose();
#pragma warning disable CS8625 // Cannot convert null literal to non-nullable reference type.
					parameter.set_grad(null);
#pragma warning restore CS8625 // Cannot convert null literal to non-nullable reference type.
				}


				//Console.WriteLine("Decoupled weight decay student word embeddings...");
				//student.DecoupledWeightDecayWordEmbeddings(studentWordEmbeddingsDecoupledWeightDecay);
				if (z == 0){
					continue;
				}
				if (z % 128 == 0)
				{
					Console.WriteLine("Saving policy...");
					string savename = save + z;
					student1.save(savename);
					savequeue.Enqueue(savename);
					if (savequeue.Count > 5)
					{
						string name = savequeue.Dequeue();
						File.Delete(name);
					}

				}

			}
		}
	}
}
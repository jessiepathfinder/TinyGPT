using Newtonsoft.Json;
using static TorchSharp.torch;
using System.Collections.Concurrent;
using System.IO.Compression;
using System.Text;
using TinyGPT.Core;
using TorchSharp;
using System.Security.Cryptography;

//TinyGPT decoder booster: catastrophic forgetting resistant single-layer perceptron
//on top of pretrained decoder attention

namespace TinyGPT.DecoderV1.Booster
{
	internal static class Program
	{
		//hyperparameters
		private const int latentTokenSize = 2048;
		private const int maxContextSize = 1025;
		private const int trainingBatches = 100000;
		private const int targetTokensPerBatch = 65536;
		private const int attentionHeads = 16;
		private const int firstTierAttentionDepth = 5;
		private const int magicTokenClasses = 4;
		private const int minimumInputTokens = 2;
		private const double regularizationTerm = 0.5;
		private const double learningRate = 1e-4;
		private const double terminalTrainingLoss = 5.0;
		//private const int maxOutputBlockSize = 128;
		private const byte maskProbability = 32;
		private const byte randomRemoveProbability = 16;
		private const byte randomVerbCorruptionProbability = 32;
		private const int regularizationLookback = 0;
		private const int startUnsupervisedTreshold = int.MaxValue;
		private const byte unsupervisedLearningRatio = 128;
		private const int wordEmbeddingUnlkTreshold = 256;
		private const int wordEmbeddingRelockTreshold = 512;
		private const int supplementalWordEmbeddingsSize = 512;
		private const double beta = 0.999;
		private const double beta_ = 1.0 - beta;
		private const double momentumDecay = 0.9;
		//private const int startUnsupervisedTreshold = 512;
		//private const double costScalingTerm = 1024.0;

		[JsonObject(MemberSerialization.Fields)]
		private sealed class WikipediaArticle
		{
			//SUPPRESS WARNINGS since fields will be reflectively set
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
#pragma warning disable CS0649
			public string title;
			public string[] section_titles;
			public string[] section_texts;
#pragma warning restore CS0649
#pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
		}

		private static void Main(string[] args)
		{
			string datadir = args[0];
			string save = args[1];
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
			Console.WriteLine("Optimizing dictionary...");
			IReadOnlyDictionary<string, OptimizedTokenizerEntry>? dict1 = Misc.OptimizeDictionary(dict);



			Console.WriteLine("Loading ELI5 + WikiQA question answering dataset...");
			Queue<string> dataqueue = new Queue<string>();
			//File.ReadAllText(datadir + "QuestionAnsweringV2.jsonl.deflate")

			using (StreamReader reader = new StreamReader(new DeflateStream(new FileStream(datadir + "QuestionAnsweringV2.jsonl.deflate", FileMode.Open, FileAccess.Read, FileShare.Read, 16777216, FileOptions.SequentialScan), CompressionMode.Decompress, false), Encoding.UTF8, false, 16777216, false))
			{
			read:
				string? line = reader.ReadLine();
				if (line is { })
				{
					dataqueue.Enqueue(line);
					goto read;
				}
			}
			string[]? questionanswering = dataqueue.ToArray();
			int wqlen2 = questionanswering.Length;

			Console.WriteLine("Starting dataset tokenizers...");
			int wqlength = wqlen2;

			ConcurrentBag<ushort[]>? alldata = new();
			//ConcurrentBag<int[]>? classcounters = new();
			int threads = Environment.ProcessorCount;
			Barrier? barrier3 = new Barrier(threads);
			int loadprogress = 0;
			Thread[] thrlist = new Thread[threads];


			for (int z = 0; z < threads; ++z)
			{
				int az = z;
				Thread thread = new Thread(() =>
				{
					int za = az;
					StringBuilder sb = new StringBuilder("Tokenized ");
					Span<ushort> encbuffer = stackalloc ushort[maxContextSize + 1];
					Span<ushort> encbuffer2 = encbuffer.Slice(1, maxContextSize);
					int mywqlen = wqlength;
					string str1 = "Tokenized {0}/" + mywqlen + " question-answer pairs";
					//int[] counter = new int[tokenClasses2];
					//classcounters.Add(counter);
					//int sa2 = suboptimalSkipInitialTokens + 2;

					while (true)
					{
						int a = Interlocked.Increment(ref loadprogress);
						if (a > mywqlen)
						{
							break;
						}
						a -= 1;
						string raw = questionanswering[a];
						if (raw[0] == '!')
						{
							raw = raw.Substring(1);
						}
						string[]? pair = JsonConvert.DeserializeObject<string[]>(raw);
						if (pair is null)
						{
							continue;
						}


						int size1 = Transformer.Tokenize(dict1, encbuffer2, pair[0], maxlen, magicTokenClasses);
						if (size1 == maxContextSize)
						{
							continue;
						}
						if (size1 < minimumInputTokens)
						{
							continue;
						}



						encbuffer2[size1++] = 0; //user-to-GPT context switch
						if (size1 == maxContextSize)
						{
							continue;
						}
						int encsize2 = size1;

						//int encsize2 = size1;
						int ctd = Transformer.Tokenize(dict1, encbuffer2[size1..], pair[1], maxlen, magicTokenClasses);
						if (ctd == 0)
						{
							continue;
						}
						size1 += ctd;
						if (size1 < maxContextSize)
						{
							encbuffer2[size1++] = 1; //GPT-to-user context switch
						}

						encbuffer[0] = (ushort)(encsize2);


						alldata.Add(encbuffer[..(size1 + 1)].ToArray());


						if ((a & 4095) == 4095)
						{
							Console.WriteLine(str1, a);
						}

					}
				});
				thread.Name = "Dataset tokenizer thread #" + z;
				thread.IsBackground = true;
				thrlist[z] = thread;
				thread.Start();
			}



			Console.WriteLine("Initializing model...");
			InitializeDeviceType(DeviceType.CUDA);
			backends.cuda.matmul.allow_tf32 = true;
			backends.cuda.matmul.allow_fp16_reduced_precision_reduction = false;
			backends.cuda.enable_math_sdp(false);
			backends.cuda.enable_flash_sdp(true);
			backends.cudnn.allow_tf32 = true;

			set_default_dtype(ScalarType.BFloat16);

			//Zero initialization makes our single-layer perceptron handle sparsity better
			//Also reduces overfitting
			Tensor booster = zeros(latentTokenSize * 2, tokenclasses, ScalarType.BFloat16, CUDA,true);
			Tensor momentum = zeros(latentTokenSize * 2, tokenclasses, ScalarType.BFloat16, CUDA,false);
			Tensor belief = zeros(latentTokenSize * 2, tokenclasses, ScalarType.BFloat16, CUDA, false);
			GPTDecoderUnitV1_1 notchatgpt = new GPTDecoderUnitV1_1("TinyGPT", latentTokenSize, attentionHeads, firstTierAttentionDepth, 0.0, 1e-7, 1024, 0.0, 1.0, 1.0, 0.0, tokenclasses, 1.0, 128, 0.0, 1, 2048, 0.0, 4);
			notchatgpt.to(bfloat16);
			notchatgpt.load(args[2]);
			notchatgpt.to(CUDA);
			Queue<string> savequeue = new Queue<string>();
			Tensor[] mergearr = new Tensor[2];
			Scalar sbeta = beta;
			Scalar slr = learningRate;
			Scalar sdecay = momentumDecay;
			Scalar sdecay_ = 1.0 - momentumDecay;
			Scalar beta_s = beta_;
			double xcomp = 1.0;
			long[] dimlist = new long[] { 1 };
			bool non_terminal = true;
			Console.WriteLine("Waiting for dataset tokenization to complete...");
			foreach (Thread thr in thrlist)
			{
				thr.Join();
			}



			Console.WriteLine("Optimizing memory usage...");
			questionanswering = null;
			dict1 = null;
			ushort[][] tokenized = alldata.ToArray();
			alldata = null;
			int datasize = tokenized.Length;
			int maxgcgen = GC.MaxGeneration;
			GC.Collect(maxgcgen, GCCollectionMode.Aggressive, true, true);
			GC.WaitForPendingFinalizers();
			for (int z = 0; z < trainingBatches; ++z){
				Console.WriteLine("Train batch #" + z);
				int tokensGenerated = 0;
				double totalLosses = 0.0;
				while(tokensGenerated < targetTokensPerBatch){
					ushort[] mydata = tokenized[RandomNumberGenerator.GetInt32(0, datasize)];
					int split = mydata[0];
					int len = mydata.Length;
					int outlen = len - split;
					tokensGenerated += outlen;
					long[] labels = new long[outlen];
					for(int i = 0; i < outlen; ++i){
						labels[i] = mydata[i + split];
					}
					Tensor x;
					using(no_grad()){
						x = notchatgpt.Forward(mydata.AsSpan(1, len - 2), split - 2, 0.0, false, true);
					}

					//ReLU v2.0 rectification helps with regularization, training stability, and anti catastrophic forgetting
					using(Tensor y = x){
						using Tensor clamped = y.clamp_max(0);
						mergearr[0] = clamped;
						mergearr[1] = y.relu_();
						x = cat(mergearr, 1);
					}
					using (Tensor y = x) x = y.matmul(booster);

					using (Tensor y = x, y1 = tensor(labels, ScalarType.Int64, CUDA)) x = Misc.FastCrossEntropyLoss(x, y1, 0.0, false);
					using(x){
						x.backward();
						totalLosses += x.ToScalar().ToDouble();
					}
				}
				double avgloss = totalLosses / tokensGenerated;
				Console.WriteLine("Average loss: " + avgloss);


				//NOTE: we use a special optimizer designed specifically for this!
				tokensGenerated *= -1;
				Console.WriteLine("Optimizer step...");
				using(no_grad()){
					Tensor grad = booster.grad() ?? throw new Exception("Booster does not have grad (should not reach here)");
					belief.mul_(sbeta);
					xcomp *= beta;
					grad.div_(tokensGenerated);

					Tensor x;
					using (Tensor y = grad.mul(grad))
					{
						x = y.mean(dimlist, true, ScalarType.Float32);
					}
					using (Tensor y = x) x = y.to(bfloat16);
					using (x)
					{
						
						belief.add_(x, beta_s);
					}

					momentum.mul_(sdecay);
					momentum.add_(grad, sdecay_);
					using (Tensor y = grad.sub(momentum)) belief.addcmul_(y, y, beta_s);
					using (Tensor y = belief.div(1.0 - xcomp))
					{
						y.sqrt_();
						booster.addcdiv_(momentum, y, slr);
					}
					grad.zero_();
				}


				if (z % 128 == 0)
				{
					Console.WriteLine("Saving policy...");
					string savename = save + z;
					booster.save(savename);
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
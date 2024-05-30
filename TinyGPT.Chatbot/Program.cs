using Newtonsoft.Json;
using System.Runtime.Serialization.Formatters;
using System.Transactions;
using TinyGPT.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Transformer = TinyGPT.Core.Transformer;

namespace TinyGPT.Chatbot
{
	internal class Program
	{
		static void Main(string[] args)
		{
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
				case "nano-v1":
					{
						const int latentTokenSize = 2048;
						maxcontext = 1025;
						const int attentionHeads = 16;
						const int firstTierAttentionDepth = 3;
						magicTokenClasses = 4;
						tokenclasses += magicTokenClasses + 1;
						themodel = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, firstTierAttentionDepth, 0.02, 128, 1e-6, 1024, 0.02, 1.0, 1.0, 0.0, tokenclasses, 6, 1.0, 2048, 0);

					}
					break;
				default:
					Console.WriteLine("unknown model!");
					return;
			}
			themodel.to(ScalarType.BFloat16);
			themodel.load(datadir + model + ".model");
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
				buffer[intokens] = 0; //[STARTGPT]
									  //int[] lastRepeat = new int[tokenclasses];
				//int[] lastRepeat = new int[tokenclasses];
				for (int i = intokens + 1, i2 = 0; i < maxcontext; ++i, ++i2)
				{
					double best = double.NegativeInfinity;
					int bestindex = -1;
					Tensor tensor;
					using (Tensor x = themodel.Forward(buffer[..i]))
					{
						tensor = x.cpu();
					}
					using (tensor)
					{
						for (int z = 0; z < tokenclasses; ++z)
						{
							double my = tensor[z].ToScalar().ToDouble();
							//my -= (1.0) / ((i-buffer[..i].LastIndexOf((ushort)z))+1);
							if (my > best)
							{
								best = my;
								bestindex = z;
							}
						}
					}

					if (bestindex == 1)
					{
						break;
					}
					//lastRepeat[bestindex] = i2;
					buffer[i] = (ushort)bestindex;
					string? str = decode[bestindex];

					if (str is null)
					{
						str = " invalid_word_" + bestindex;
					}
					Console.Write(str);
				}
				Console.WriteLine();
				Console.WriteLine("==================================================");
				Console.WriteLine();

			}
		}
	}
}
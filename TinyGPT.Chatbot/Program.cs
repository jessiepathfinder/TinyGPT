﻿using Newtonsoft.Json;
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
	internal static class Program
	{
		static void Main(string[] args)
		{
			string? temperature_str = Environment.GetEnvironmentVariable("TinyGPT_generation_temperature");

			double temperature = temperature_str is null ? 0.5 : Convert.ToDouble(temperature_str);
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
				case "nano-v2":
					{
						const int latentTokenSize = 2048;
						maxcontext = 1025;
						const int attentionHeads = 16;
						const int firstTierAttentionDepth = 5;
						magicTokenClasses = 4;
						tokenclasses += magicTokenClasses + 1;
						themodel = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, firstTierAttentionDepth, 0.02, 1e-6, 1024, 0.02, 1.0, 1.0, 0.0, tokenclasses, 1.0, 128, 0.0, 1, 2048, 0.0);

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
			Span<float> topk = stackalloc float[256];
			Span<ushort> taboo = stackalloc ushort[4];
			int tkwindow = 0;
			Misc.MakeSecureRandomFloats(topk);
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
				for(int i = 0; i < 4; ){
					taboo[i++] = 65535;
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
					
					
					Tensor tensor, indices;
					using (Tensor x = themodel.Forward(buffer[..i]))
					{
						(tensor, indices) = x.sort(descending: true);
					}
					using (Tensor x = tensor)
					{
						tensor = x.to(float64);
					}
					using (Tensor x = tensor)
					{
						tensor = x.softmax(0);
					}
					using (Tensor x = tensor)
					{
						tensor = x.cpu();
					}
					double tk = 1.0;
					//MASK taboo words
					for(int p = 0; p < 4; ){
						ushort myr = taboo[p++];
						if (myr == 65535) continue;
						string? detokenized = decode[myr];
						if (detokenized is null)
							continue;
						if (!detokenized.Contains(','))
							continue;
						Scalar sc;
						using (Tensor x = tensor[myr]) sc = x.ToScalar();
						tk -= sc.ToDouble();
						tensor[myr] = 0.0;

					}
					tk *= topk[tkwindow] * temperature;
					if (tkwindow == 255)
					{
						tkwindow = 0;
						Misc.MakeSecureRandomFloats(topk);
					}
					else
					{
						++tkwindow;
					}
					ushort bestindex;
					using (indices)
					{
						int z = 0;
						using (tensor){
							
							for (; z < tokenclasses & tk > 0.0; ++z)
							{
								using Tensor tt = tensor[z];
								tk -= tt.ToScalar().ToDouble();
							}
						}
						using Tensor tt2 = indices[z];
						bestindex = (ushort)tt2.ToScalar().ToInt32();
					}
					

					if (bestindex == 1)
					{
						break;
					}
					//lastRepeat[bestindex] = i2;
					taboo[i % 4] = bestindex;
					buffer[i] = bestindex;
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
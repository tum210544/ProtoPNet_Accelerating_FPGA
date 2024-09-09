import math
def estimate_2loop(out_channel, out_size, in_channel, in_kernel, inner_unroll, out_unroll, in_laten, in_ii, out_laten, out_ii):
    out_iter = out_channel * out_size * out_size
    out_batch = math.ceil(out_iter/out_unroll)
    in_iter = in_channel * in_kernel * in_kernel
    in_batch = math.ceil(in_iter/inner_unroll)
    all_cycle = out_batch*in_batch*in_ii + in_laten + out_laten
    return all_cycle

inner_unroll = 1
inner_latency = 150
inner_ii = 1

out_unroll = 1
out_latency = 128
out_ii = 1

freq = 299.22

conv1 = estimate_2loop(64,224,3,3,inner_unroll, out_unroll, inner_latency,inner_ii, out_latency, out_ii)
conv2 = estimate_2loop(64,224,64,3,inner_unroll, out_unroll, inner_latency,inner_ii, out_latency, out_ii)
conv3 = estimate_2loop(128,112,64,3,inner_unroll, out_unroll, inner_latency,inner_ii, out_latency, out_ii)
conv4 = estimate_2loop(128,112,128,3,inner_unroll, out_unroll, inner_latency,inner_ii, out_latency, out_ii)
conv5 = estimate_2loop(256,56,128,3,inner_unroll, out_unroll, inner_latency,inner_ii, out_latency, out_ii)
conv6_8 = estimate_2loop(256,56,256,3,inner_unroll, out_unroll, inner_latency,inner_ii, out_latency, out_ii) * 3
conv9 = estimate_2loop(512,28,256,3,inner_unroll, out_unroll, inner_latency,inner_ii, out_latency, out_ii)
conv10_12 = estimate_2loop(512,28,512,3,inner_unroll, out_unroll, inner_latency,inner_ii, out_latency, out_ii) * 3
conv13_16 = estimate_2loop(512,14,512,3,inner_unroll, out_unroll, inner_latency,inner_ii, out_latency, out_ii) * 4
conv17 = estimate_2loop(128,7,512,1,inner_unroll, out_unroll, inner_latency,inner_ii, out_latency, out_ii)
conv18= estimate_2loop(128,7,128,1,inner_unroll, out_unroll, inner_latency,inner_ii, out_latency, out_ii)

total = conv1+conv2+conv3+conv4+conv5+conv6_8+conv9+conv10_12+conv13_16+conv17+conv18
total_duration = (total/1000000)/freq

print(total)
print(total_duration)

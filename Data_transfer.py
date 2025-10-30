import numpy as np
import matplotlib.pyplot as plt
import input
from helper_functions import *
import re

# this file contains the queueing models needed to simulate data transfer through the paths selected during routing
# currently a queueing model with general arrival distribution, deterministic processing and a standard buffer size is implemented
# packet processing can be done on sequential or pipeline basis, see below for elaboration

def GDcK_queueing(link_performance_dicts,
                  routing_source_destination,
                  routing_time,
                  routing_nodes,
                  routing_edges,
                  buffer_capacity=np.inf,
                  packet_processing='sequential',
                  plot=False):

    # create buffer designations per time stamp
    buffers_t = []
    for t in range(len(routing_edges)):
        route_buffers = []
        for n in range(len(routing_edges[t])):
            current_route = routing_edges[t][n]
            current_route_buffers = []
            if len(current_route) > 1:
                for m in range(1, len(current_route)):
                    current_edge = list(current_route[m - 1])
                    next_edge = list(current_route[m])
                    nodes = list(current_edge)
                    nodes.append(next_edge[1])
                    node_parts = [node.replace('V_', '') for node in nodes]
                    buffer_name = 'b_' + '_'.join(node_parts)
                    current_route_buffers.append(buffer_name)
            else:
                current_edge = list(current_route[0])
                nodes = list(current_edge)
                nodes.append(current_edge[1]) # create a buffer name where the edge is duplicated to show there was only one link in the route
                node_parts = [node.replace('V_', '') for node in nodes]
                buffer_name = 'b_' + '_'.join(node_parts)
                current_route_buffers.append(buffer_name)
            route_buffers.append(current_route_buffers)
        buffers_t.append(route_buffers)

    # create unique buffer sets for every route
    buffer_sets = []
    for n in range(len(routing_source_destination)):
        all_route_buffers = []
        for t in range(len(buffers_t)):
            for route_buffer in buffers_t[t][n]:
                all_route_buffers.append(route_buffer)
        route_buffer_set = duplicate_filter_with_order(all_route_buffers)
        buffer_sets.append(route_buffer_set)

    # create corresponding edge sets
    edge_sets = []
    for n in range(len(buffer_sets)):
        route_edges = []
        for buffer_name in buffer_sets[n]:
            buffer_nums = re.findall(r'\d+', buffer_name)
            buffer_nodes = [(buffer_nums[n], buffer_nums[n+1]) for n in range(0, len(buffer_nums), 2)]
            if buffer_nodes[1] == buffer_nodes[2]: # in case there is only one link per route
                nodes = [
                    'V_' + str(buffer_nodes[0][0]) + '_' + str(buffer_nodes[0][1]),
                    'V_' + str(buffer_nodes[1][0]) + '_' + str(buffer_nodes[1][1])
                ]
                edges = [(nodes[0], nodes[1])] * 2
            else:
                nodes = ['V_' + str(buffer_nodes[i][0]) + '_' + str(buffer_nodes[i][1]) for i in range(len(buffer_nodes))]
                edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

            # parts = buffer_name.split('_')[1:]  # remove the 'b' and split the nodes
            # nodes = ['V_' + parts[i] + '_' + parts[i + 1] for i in range(0, len(parts), 2)]
            # edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
            route_edges.append(edges)
        edge_sets.append(route_edges)



    queueing_output_total = []
    # find out if both buffer edges were active during a specific time stamp for a route
    for n in range(len(routing_source_destination)):

        ##### with buffer drain #####
        queueing_output = {
            'source destination': routing_source_destination[n],
            'edges': edge_sets[n],
            'buffers': buffer_sets[n],
            'time': [[] for _ in range(len(buffer_sets[n]))],
            'effective throughput': [[] for _ in range(len(buffer_sets[n]))],
            'buildup': [[] for _ in range(len(buffer_sets[n]))],
            'transfer': [[] for _ in range(len(buffer_sets[n]))],
            'occupancy': [[] for _ in range(len(buffer_sets[n]))],
            'drop count': [[] for _ in range(len(buffer_sets[n]))],
            'queue time': [[] for _ in range(len(buffer_sets[n]))],
            'node latency': [[] for _ in range(len(buffer_sets[n]))],
            'propagation latency': [[] for _ in range(len(buffer_sets[n]))]
        }

        route_link_times = link_performance_dicts[n]['time']
        route_link_names = link_performance_dicts[n]['link name']
        route_link_throughput = link_performance_dicts[n]['throughput']
        route_link_ranges = link_performance_dicts[n]['ranges']

        for t in range(len(routing_time)):
            if len(routing_edges[t][n]) > 1:
                current_edge_pairs = [[routing_edges[t][n][i-1], routing_edges[t][n][i]] for i in range(1, len(routing_edges[t][n]))]
            else:
                current_edge_pairs = [[routing_edges[t][n][0], routing_edges[t][n][0]]]
                # print('----------------')
                # print(current_edge_pairs)

            # print('current edge pairs:', current_edge_pairs)
            prev_buffer_occupancies = []
            current_throughput_pairs = []

            for j in range(len(current_edge_pairs)):
                current_edge_0 = current_edge_pairs[j][0]
                current_edge_1 = current_edge_pairs[j][1]

                # select the corresponding indices in the link performance dict
                # (for loop necessary because the edge entries are not arrays)
                edge_indices = np.empty(2)
                for k in range(len(route_link_names)):
                    if route_link_names[k] == current_edge_0:
                        edge_indices[0] = k
                    if route_link_names[k] == current_edge_1:
                        edge_indices[1] = k

                # print('edge indices:', edge_indices)

                edge_index_0 = int(edge_indices[0])
                edge_index_1 = int(edge_indices[1])

                time_index_0 = np.where(route_link_times[edge_index_0] == routing_time[t])[0][0]
                time_index_1 = np.where(route_link_times[edge_index_1] == routing_time[t])[0][0]

                current_throughput_0 = route_link_throughput[edge_index_0][time_index_0]
                current_throughput_1 = route_link_throughput[edge_index_1][time_index_1]

                current_range_0 = route_link_ranges[edge_index_0][time_index_0]
                current_range_1 = route_link_ranges[edge_index_1][time_index_1]

                # select the corresponding buffer index to extract the previous occupancy (using the current edge pair and the edge sets)
                for i in range(len(edge_sets[n])):
                    if edge_sets[n][i] == current_edge_pairs[j]:
                        current_buffer_idx = i

                # print('buffer index:', current_buffer_idx)

                buffer_occupancy_history = queueing_output['occupancy'][current_buffer_idx]
                if len(buffer_occupancy_history) > 0:
                    prev_buffer_occupancy = buffer_occupancy_history[-1]
                else:
                    prev_buffer_occupancy = 0.0

                prev_buffer_occupancies.append(prev_buffer_occupancy)

                # throughput logic (two if with an implicit else that throughput will be same as in MMM)
                # base the current inbound throughput on the outbound throughput of the previous buffer
                if len(current_throughput_pairs) > 0:
                    current_throughput_0 = current_throughput_pairs[-1][1] # select the outbound throughput of the previous buffer
                # special case when the buffer is empty and the outgoing throughput is bigger then the incoming one
                # this means the output will be limited to input
                # in the other cases the buffer will either be filled (in > out) or drained (buffer > 0.0 and out > in)
                if prev_buffer_occupancy == 0.0 and current_throughput_1 > current_throughput_0:
                    current_throughput_1 = current_throughput_0

                if packet_processing == 'sequential':
                    # effective throughput when accounting for processing time in sat electrical system (package handling)
                    # this has to be done for both arrival and departure rates
                    # in the 'sequential case', the packet experiences both the processing time and transmission time
                    # this comes from the assumption that the switching hardware is not able to perform parallel operations
                    # arrival logic
                    arrival_rate = 1 / (input.packet_size / current_throughput_0 + input.latency_processing) if current_throughput_0 > 0.0 else 0.0
                    # departure logic
                    # the slowest of the two rates determines the final departure rate
                    departure_rate = 1 / (input.packet_size / current_throughput_1 + input.latency_processing) if current_throughput_1 > 0.0 else 0.0

                if packet_processing == 'pipeline':
                    # in this case the switching hardware is able to perform parallel operations
                    # this leads to either the transmission time or processing time being the bottleneck, depending on which is bigger
                    # this means that for example a packet can already start processing before the current packet has been transmitted when Ttrans > Tprocess
                    arrival_rate = min(1 / input.latency_processing, current_throughput_0 / input.packet_size)
                    # departure logic
                    departure_rate = min(1 / input.latency_processing, current_throughput_1 / input.packet_size)

                # correct the departure rate if the buffer gets drained to 'below zero' in between time steps
                if prev_buffer_occupancy > 0.0 and departure_rate > arrival_rate:
                    drain_rate = departure_rate - arrival_rate
                    t_drain = prev_buffer_occupancy / drain_rate
                    # print('---------')
                    # print('buffer name:', queueing_output['buffers'][current_buffer_idx])
                    # print('prev buffer occ:', prev_buffer_occupancy)
                    # print('drain time:', t_drain)
                    # print('arr/dep:', arrival_rate, departure_rate)
                    if t_drain < input.step_size_link:
                        average_departure_rate = (t_drain * departure_rate + (input.step_size_link - t_drain) * arrival_rate) / input.step_size_link
                        rate_ratio = average_departure_rate / departure_rate

                        current_throughput_1 = current_throughput_1 * rate_ratio
                        departure_rate = average_departure_rate

                        # print(rate_ratio)
                        # print('New dep rate:', departure_rate)
                        # print('drain time new rates:', prev_buffer_occupancy / (departure_rate - arrival_rate))
                current_throughput_pairs.append([current_throughput_0, current_throughput_1])

                # t_offset to account for propagation in arrival of packets
                # also done for departure of packets (especially for the arrival of packets at the last node)
                # probably not necessary because packets from previous time slot that did not reach node yet will arrive first
                # leading to the full time step worth of data anyways
                # t_offset_arr = input.step_size_link - current_range_0 / input.speed_of_light
                # t_offset_dep = input.step_size_link - current_range_1 / input.speed_of_light

                # an offset step size is created to account for the the propagation time of a packet. In this way only the packets that arrive within the simulation step size are accounted for
                buildup = (arrival_rate - departure_rate) * input.step_size_link #* t_offset_arr  # alternatively just use input.step_size_link if the propagation time is smaller then the simulation step size
                occupancy_unlimited = max(0.0, prev_buffer_occupancy + buildup)
                occupancy = min(occupancy_unlimited, buffer_capacity / input.packet_size)
                drop = max(0.0, occupancy_unlimited - occupancy)
                transfer = departure_rate * input.step_size_link #* t_offset_dep
                t_queue = occupancy / departure_rate if departure_rate > 0.0 else occupancy * input.latency_processing
                t_node = t_queue + 1 / departure_rate if departure_rate > 0.0 else t_queue + input.latency_processing
                t_prop = np.array([current_range_0, current_range_1]) / input.speed_of_light

                queueing_output['time'][current_buffer_idx].append(routing_time[t])
                queueing_output['effective throughput'][current_buffer_idx].append([arrival_rate, departure_rate])
                queueing_output['buildup'][current_buffer_idx].append(buildup)
                queueing_output['occupancy'][current_buffer_idx].append(occupancy)
                queueing_output['drop count'][current_buffer_idx].append(drop)
                queueing_output['transfer'][current_buffer_idx].append(transfer)
                queueing_output['queue time'][current_buffer_idx].append(t_queue)
                queueing_output['node latency'][current_buffer_idx].append(t_node)
                queueing_output['propagation latency'][current_buffer_idx].append(t_prop)

                # if current_edge_pairs[0][0] == current_edge_pairs[0][1]:
                #     print(current_edge_pairs, queueing_output['buffers'][current_buffer_idx], buffer_sets[n][current_buffer_idx], edge_sets[n][current_buffer_idx], queueing_output['time'][current_buffer_idx], routing_time[t], current_buffer_idx)

        queueing_output_total.append(queueing_output)

    if plot:
        for n in range(len(queueing_output_total)):
            src_dest = queueing_output_total[n]['source destination']
            names = queueing_output_total[n]['buffers']
            times = queueing_output_total[n]['time']
            build = queueing_output_total[n]['buildup']
            occupancy = queueing_output_total[n]['occupancy']
            queues = queueing_output_total[n]['queue time']
            transfer = queueing_output_total[n]['transfer']

            if len(names) == 1:
                # set font size
                plt.rcParams.update({'font.size': 12})

                fig, ax = plt.subplots(1, figsize=(6, 4))
                plt.suptitle(f'Route {n} buffer occupancy {src_dest}')
                # ax[plot_ctr].bar(times[m], build[m]) # buildup rate
                ax.bar(times[0], occupancy[0])  # bar
                ax.plot(times[0], occupancy[0], color='red') # line
                ax.set_ylabel(f'{names[0]} \n occupancy [packets]')

                # ax[plot_ctr].bar(times[m], queues[m])
                # ax[plot_ctr].set_ylabel(f'{names[m]} \n queue time [s]', rotation=0, labelpad=50)

                ax.set_xlabel('time [s]')
                ax.set_xlim(input.start_time, input.end_time)

                plt.tight_layout()

                plt.savefig(f'figures/BufferOccupancy/buffer_buildup_plots_path_{n}_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')

            # in this case only buffers with queued packets are shown
            if len(names) > 1:
                # first select the buffers that are occupied
                occupied_buffers_indices = []
                for i in range(len(names)):
                    if np.sum(occupancy[i]) > 0:
                        occupied_buffers_indices.append(i)

                if len(occupied_buffers_indices) == 1:
                    idx = occupied_buffers_indices[0]
                    fig, axs = plt.subplots(1, figsize=(12, 12))
                    plt.suptitle(f'Route {n} buffer occupancy {src_dest}')
                    # axs[plot_ctr].bar(times[m], build[m]) # buildup rate
                    axs.bar(times[idx], occupancy[idx])  # buffer occupancy
                    axs.set_ylabel(f'{names[idx]} \n occupancy [packets]', rotation=0, labelpad=50)

                    # axs.bar(times[m], queues[m])
                    # axs.set_ylabel(f'{names[m]} \n queue time [s]', rotation=0, labelpad=50)

                    axs.set_xlabel('time [s]')
                    axs.set_xlim(input.start_time, input.end_time)

                    plt.tight_layout()

                    plt.savefig(f'figures/BufferOccupancy/buffer_buildup_plots_path_{n}_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')

                if len(occupied_buffers_indices) > 1:
                    fig, axs = plt.subplots(len(occupied_buffers_indices), figsize=(12, 12))
                    plt.suptitle(f'Route {n} buffer occupancy {src_dest}')
                    plot_ctr = 0
                    for idx in occupied_buffers_indices:
                        # axs[plot_ctr].bar(times[m], build[m]) # buildup rate
                        axs[plot_ctr].bar(times[idx], occupancy[idx]) # buffer occupancy
                        axs[plot_ctr].set_ylabel(f'{names[idx]} \n occupancy [packets]', rotation=0, labelpad=50)

                        # axs[plot_ctr].bar(times[m], queues[m])
                        # axs[plot_ctr].set_ylabel(f'{names[m]} \n queue time [s]', rotation=0, labelpad=50)

                        axs[plot_ctr].set_xlabel('time [s]')
                        axs[plot_ctr].set_xlim(input.start_time, input.end_time)

                        plot_ctr += 1

                    plt.tight_layout()

                    plt.savefig(f'figures/BufferOccupancy/buffer_buildup_plots_path_{n}_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')

        # # path latency plot
        # plt.figure()
        # plt.title('Path latencies over time for every route')
        # for m in range(len(queueing_output_total)):
        #     plt.plot(routing_time, np.asarray(queueing_output_total[m]['path latency']) * 1e3, label=f'route {m}')
        # plt.ylabel('latency [ms]')
        # plt.xlabel('time [s]')
        # plt.yscale('log')
        # plt.legend()
        #
        # plt.savefig(f'figures/path_latency_plots_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')
        #
        # fig, ax = plt.subplots()
        # ax.set_title('Cumulative received data for every route')
        # for m in range(len(queueing_output_total)):
        #     ax.plot(routing_time, queueing_output_total[m]['cum received data'] * 1e-9, label=f'route {m} received data', linestyle='--')
        #     # ax.plot(routing_time, np.asarray(queueing_output_total[m]['received data']) * 1e-9, label=f'route {m}')
        # ax.set_ylabel('data [Gb]')
        # ax.set_xlabel('time [s]')
        # ax.legend()
        #
        # ax1 = ax.twinx()
        # for m in range(len(queueing_output_total)):
        #     ax1.plot(routing_time, queueing_output_total[m]['path throughput'] * 1e-9, label=f'route {m} path throughput')
        # ax1.set_ylabel('data [Gb/s]')
        # ax1.set_xlabel('time [s]')
        # ax1.legend()
        #
        # plt.savefig(f'figures/data_received_plots_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')

        plt.show()

    return queueing_output_total
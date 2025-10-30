import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# import the constellation mission function
from constellation_mission_level import *

# color list
import matplotlib.colors as mcolors
color_list = list(dict(mcolors.TABLEAU_COLORS).keys())
color_list_2 = list(dict(mcolors.CSS4_COLORS).keys())

# with this script the final analysis of the constellations can be performed
# There are options to retrieve pickles figures, perform a convergence test for the platform vibrations,
# sensitivity analysis, direct constellation comparison or a literature comparison

# create constellation comparison folder
comparison_path = 'figures/ConstellationComparison'
os.makedirs(comparison_path, exist_ok=True)

# script logic
retrieve_figures = False
convergence_test = False
single_test = True
analyze_param_var = False
analyze_custom_constellation_collection = False
analyze_latency_literature = False

# retrieve pickles
if retrieve_figures:
    def retrieve_pickled_plot(path):
        with open(path, 'rb') as file:
            fig = pickle.load(file)
        return fig

    pickle_path = "Constellation_results/i88_T105_P7_F2_h780km/Resolite80_2-5Gbps_4LCTs/dijkstra_3routes_sim_540s_nl/figures/pickles/routing_P7_S15.pkl"

    fig_0 = retrieve_pickled_plot(pickle_path)
    plt.show()

    exit()

# create the set of constellations that will be analyzed
performance_variables = ['Files', 'Efficiency', 'Users', 'Latency', 'Slew', 'Reliability', 'Compound']
param_names = ['Inclination', 'Sats per plane', 'Planes', 'Phasing', 'Altitude', 'Terminals']
param_names_short = [r'$i$', r'$S$', r'$P$', r'$F$', r'$h$', r'$N_\text{LCT}$']
param_units = ['[deg]', '[-]', '[-]', '[-]', '[km]', '[-]']
default_params = np.array([65, 10, 10, 5, 1200.0E3, 4])
# elaborate
i_range = np.array([55, 60, 70, 75, 80, 85])
S_range = np.array([8, 9, 11, 12, 13, 14])
P_range = np.array([8, 9, 11, 12, 13, 14])
F_range = np.array([2, 3, 4, 6, 7, 8])
h_range = np.array([900.0E3, 1000.0E3, 1100.0E3, 1300.0E3, 1400.0E3, 1500.0E3])
lct_range = np.array([3])

if convergence_test:
    # set font size
    plt.rcParams.update({'font.size': 13})

    params = np.array([65, 70, 5, 1, 2000.0E3, 4])  # convergence analysis wdc
    # params = np.array([65, 9, 10, 5, 1200.0E3, 4])  # second to last for worst performing sensitivity analysis constellations

    cons_name = 'Convergence 4'

    # simulation time fraction of an hour
    t_frac = 1.0

    # number of routes
    number_routes = 1

    # step collection
    link_steps = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    # link_steps = [1.0]

    # convergence_results
    range_results = []
    P_r_results = []
    frac_fade_results = []
    mean_fade_results = []
    number_fade_results = []

    print('Beginning convergence analysis...')
    for link_step in link_steps:
        print('--------------------')
        print(f'Current link step = {link_step} s')
        # run the constellation mission file to simulate topology, routing and OISL's
        result, link_result, path_result = constellation_mission(
            name=cons_name,
            inclination=params[0],
            sats_total=int(params[1] * params[2]),
            planes=int(params[2]),
            sats_per_plane=int(params[1]),
            phasing=int(params[3]),
            altitude=params[4],
            lcts=int(params[5]),
            number_routes=number_routes,
            t_frac=t_frac,
            link_step=link_step,
            ext_nodes=False,        # enable terrestrial nodes
            link_simulation=True,   # enable link simulation
            shared_links=False,     # enable link capacity
            plots=False,
            verification=True
        )

        # the convergence analysis is performed on the first inter-orbit link of a constellation (due to it variance over time)
        # the information used are the fade statistics and the average received powr
        for n in range(len(link_result)):
            for m in range(len(link_result[n]['link name'])):
                link_val = link_result[n]['link name'][m]
                if link_val == ('V_0_0', 'V_1_0'):
                    # time_vals = link_result[n]['time'][m]
                    range_vals = link_result[n]['ranges'][m] * 1e-3
                    P_r_vals = W2dB(link_result[n]['Pr mean'][m])
                    frac_fade_vals = link_result[n]['fractional fade time'][m]
                    mean_fade_vals = link_result[n]['mean fade time'][m] * 1e3
                    number_fade_vals = link_result[n]['number of fades'][m]

                    # If the simulation time is longer than 0.25hr, the relative states of will be very similar at some points
                    # therefore the link ranges are sorted to get ordered plots
                    idc_sorted = range_vals.argsort()
                    range_sorted = range_vals[idc_sorted]
                    P_r_sorted = P_r_vals[idc_sorted]
                    frac_fade_sorted = frac_fade_vals[idc_sorted]
                    mean_fade_sorted = mean_fade_vals[idc_sorted]
                    number_fade_sorted = number_fade_vals[idc_sorted]

                    # append to results lists
                    range_results.append(range_sorted)
                    P_r_results.append(P_r_sorted)
                    frac_fade_results.append(frac_fade_sorted)
                    mean_fade_results.append(mean_fade_sorted)
                    number_fade_results.append(number_fade_sorted)

    print('Plotting results...')
    # plot power and fades vs link range
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    plt.suptitle(f'Inter-orbit link convergence analysis for {input.inc_SC}:{input.total_number_of_sats}/{input.number_of_planes}/{input.phasing_factor} at {input.h_SC * 1e-3}km' + r' ($\Delta T_{micro}$ = 0.5 ms)')

    for n in range(len(link_steps)):
        ax0.plot(range_results[n], P_r_results[n], label=r'$\Delta T_{macro} = $' + f'{int(link_steps[n])} s') # power
        ax1.plot(range_results[n], frac_fade_results[n], label=r'$\Delta T_{macro} = $' + f'{int(link_steps[n])} s') # frac fade time
        ax2.plot(range_results[n], mean_fade_results[n], label=r'$\Delta T_{macro} = $' + f'{int(link_steps[n])} s') # mean fade time

    ax0.set_ylabel(r'$P_{RX}$ [dB]')
    ax1.set_ylabel('Frac fade\n time [-]')
    ax2.set_ylabel('Mean fade\n time [ms]')
    ax2.set_xlabel('Link range [km]')

    ax1.set_yscale('log')
    ax2.set_yscale('log')

    ax0.grid()
    ax1.grid()
    ax2.grid()

    ax0.legend(loc='upper right', ncol=3)

    plt.tight_layout()

    # save png
    image_path = f'figures/Pointing/S{int(params[1] * params[2])}_P{int(params[2])}convergence_power_fades.png'
    plt.savefig(image_path, dpi=125)
    # save pickle
    pickle_file_path = image_path.replace('png', 'pkl')
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(fig, file)
    pickle_plot(pickle_file_path, fig)

    plt.show()

if single_test:

    # constellation input parameters
    params = [88, 15, 7, 2, 780.0E3, 4]

    # constellation name
    cons_name = 'Custom 4'

    # simulation time fraction of an hour
    t_frac = 1.5

    # number of routes
    number_routes = 2

    # link simulation step
    link_step = 5.0

    # run the constellation mission file to simulate topology, routing and OISL's
    result, link_result, path_result = constellation_mission(
        name=cons_name,
        inclination=params[0],
        sats_total=int(params[1] * params[2]),
        planes=int(params[2]),
        sats_per_plane=int(params[1]),
        phasing=int(params[3]),
        altitude=params[4],
        lcts=int(params[5]),
        number_routes=number_routes,
        t_frac=t_frac,
        link_step=link_step,
        ext_nodes=False,
        link_simulation=False,
        shared_links=False,
        verification=False,
        RF_latency=False,
        availability_sun=False
    )

    # print('Link')
    # print(link_result)

    # print('Path')
    # print(path_result)
    # path verification
    # random.seed(1337)
    # for n in range(number_routes):
    #     # select
    #     path_edges = path_result[n]['path edges']
    #     random_indices = sorted(np.array([0] + [random.randint(1, len(path_edges)) for _ in range(4)]))
    #     path_selection = path_edges[random_indices]
    #     print('--------')
    #     print('indices', random_indices)
    #     print(f'route {n}')
    #     print(np.vstack(path_selection))


    print('--------')
    print('Results:')
    print(result)

    # print(path_result)

    # print('----------')
    # print('Results reading check')
    # result_check = read_txt(loc)
    # print(result_check)

    files = result['transferred files']
    efficiency = result['path efficiency']
    users = result['user ratio']
    latency = result['path latency']
    slew = result['slew rate']
    reliability = result['reliability']
    compound = result['compound']

    plt.figure()
    plt.title(f'{params[0]}:{params[1] * params[2]}/{params[2]}/{params[3]} \n'
              f'at {int(params[4] * 1e-3)}km with {int(params[5])} LCTs')
    # plt.scatter(0, files ,label='transferred file', s=10)
    # plt.scatter(1,efficiency ,label='path efficiency' ,s=10)
    # plt.scatter(2, users, label ='user ratio', s=10)
    # plt.scatter(3, latency, label ='path latency', s=10)
    # plt.scatter(4, slew, label ='slew rates', s=10)
    # plt.scatter(5, reliability, label ='reliability', s=10)
    # plt.scatter(6, compound, label='compound', s=10)
    plt.bar(0, files ,label='transferred file', width=0.5)
    plt.bar(1,efficiency ,label='path efficiency', width=0.5)
    plt.bar(2, users, label ='user ratio', width=0.5)
    plt.bar(3, latency, label ='path latency', width=0.5)
    plt.bar(4, slew, label ='slew rates', width=0.5)
    plt.bar(5, reliability, label ='reliability', width=0.5)
    plt.bar(6, compound, label='compound', width=0.5)
    plt.xticks(range(7), performance_variables, rotation=45)
    plt.xlabel('parameters')
    plt.ylabel('Performance norm [-]')
    # plt.legend()
    # plt.grid()
    plt.tight_layout()
    plt.savefig(f'figures/ConstellationComparison/single_test_inc{params[0]}_T{params[1] * params[2]}_P{params[2]}_F{params[3]}_h{int(params[4] * 1e-3)}km_{int(params[5])}LCTs.png')
    plt.show()


# analyze all parameters
if analyze_param_var:

    # set font size
    plt.rcParams.update({'font.size': 17})

    constellation_set = [[] for _ in range(len(default_params))]
    constellation_set.append([default_params])

    for i in i_range:
        current_constellation = default_params.copy()
        current_constellation[0] = i
        constellation_set[0].append(current_constellation)
    for S in S_range:
        current_constellation = default_params.copy()
        current_constellation[1] = S
        constellation_set[1].append(current_constellation)
    for P in P_range:
        current_constellation = default_params.copy()
        current_constellation[2] = P
        constellation_set[2].append(current_constellation)
    for F in F_range:
        current_constellation = default_params.copy()
        current_constellation[3] = F
        constellation_set[3].append(current_constellation)
    for h in h_range:
        current_constellation = default_params.copy()
        current_constellation[4] = h
        constellation_set[4].append(current_constellation)
    for lct in lct_range:
        current_constellation = default_params.copy()
        current_constellation[5] = lct
        constellation_set[5].append(current_constellation)

    # get the default constellation result
    print('----------------')
    print(f'DEFAULT constellation configuration is [i, S, P, F, h, LCT]:')
    print(default_params)

    # simulation time
    t_frac = 1.0

    # number of routes
    number_routes = 8

    # set the constellation configurations
    default_result, default_link_result, default_path_result = constellation_mission(
        name=f'constellation_default',
        inclination=default_params[0],
        sats_total=int(default_params[1] * default_params[2]),
        planes=int(default_params[2]),
        sats_per_plane=int(default_params[1]),
        phasing=int(default_params[3]),
        altitude=default_params[4],
        lcts=int(default_params[5]),
        number_routes=number_routes,
        t_frac=t_frac
    )
    plt.close()

    multi_var = True

    if multi_var:
        analysis_indices = [0, 1, 2, 3, 4, 5] # select which parameters you want to vary
        # analysis_indices = [0, 1, 2]
        indices = [str(x) for x in analysis_indices]
        save_indices = '_'.join(indices)
        print('multiple parameters will be analyzed:', save_indices)
    else:
        analysis_indices = [2]  # corresponds to the index of to be analyzed parameters influence
        save_indices = analysis_indices[0]
        print('A single parameter will be analyzed:', save_indices)

    x_plusminus = []
    x_plusminus_collection = []
    compounds = []
    compound_names = []
    for n in analysis_indices:
        files = []
        efficiency = []
        users = []
        latency = []
        slew = []
        reliability = []
        compound = []
        for m in range(len(constellation_set[n])):
            print('----------------')
            print(f'Constellation {m} configuration is [i, S, P, F, h, LCT]:')
            print(constellation_set[n][m])

            # set the constellation configurations
            inclination = constellation_set[n][m][0]
            sats_total = constellation_set[n][m][1] * constellation_set[n][m][2]
            planes = constellation_set[n][m][2]
            sats_per_plane = constellation_set[n][m][1]
            phasing = constellation_set[n][m][3]
            altitude = constellation_set[n][m][4]
            lcts = constellation_set[n][m][5]

            # run the constellation mission file to simulate topology, routing and OISL's
            result, link_result, path_result = constellation_mission(
                name=f'constellation_{m}',
                inclination=inclination,
                sats_total=int(sats_total),
                planes=int(planes),
                sats_per_plane=int(sats_per_plane),
                phasing=int(phasing),
                altitude=altitude,
                lcts=int(lcts),
                number_routes=number_routes,
                t_frac=t_frac
            )
            plt.close()

            files.append(result['transferred files'])
            efficiency.append(result['path efficiency'])
            users.append(result['user ratio'])
            latency.append(result['path latency'])
            slew.append(result['slew rate'])
            reliability.append(result['reliability'])
            compound.append(result['compound'])

            print('--------')
            print('Results:')
            print(result)

        # append default result
        files.append(default_result['transferred files'])
        efficiency.append(default_result['path efficiency'])
        users.append(default_result['user ratio'])
        latency.append(default_result['path latency'])
        slew.append(default_result['slew rate'])
        reliability.append(default_result['reliability'])
        compound.append(default_result['compound'])

        # plot the combined results per metric
        default_x_param = constellation_set[-1][0][n]
        default_compound = compound[-1]

        x_params = [constellation_set[n][i][n] for i in range(len(constellation_set[n]))]
        x_params.append(constellation_set[-1][0][n]) # append default param

        indices_x = sorted(enumerate(x_params), key=lambda i: i[1])
        x_params = [x_params[idx_val[0]] for idx_val in indices_x]
        files = [files[idx_val[0]] for idx_val in indices_x]
        efficiency = [efficiency[idx_val[0]] for idx_val in indices_x]
        users = [users[idx_val[0]] for idx_val in indices_x]
        latency = [latency[idx_val[0]] for idx_val in indices_x]
        slew = [slew[idx_val[0]] for idx_val in indices_x]
        reliability = [reliability[idx_val[0]] for idx_val in indices_x]
        compound = [compound[idx_val[0]] for idx_val in indices_x]

        plt.figure(figsize=(8, 7))
        # plt.title(f'{param_names[n]} variation \n'
        #           f'{default_params[0]}:{default_params[1] * default_params[2]}/{default_params[2]}/{default_params[3]} \n'
        #           f'at {int(default_params[4] * 1e-3)}km with {default_params[5]} LCTs')
        plt.title(f'{param_names[n]} variation individual performance')
        plt.plot(x_params, files, marker='o', label=r'$\mathcal{M}_\text{D}$')
        plt.plot(x_params, efficiency, marker='o', label=r'$\mathcal{M}_\eta$')
        plt.plot(x_params, users, marker='o', label=r'$\mathcal{M}_\text{U}$')
        plt.plot(x_params, latency, marker='o', label=r'$\mathcal{M}_\tau$')
        plt.plot(x_params, slew, marker='o', label=r'$\mathcal{M}_\omega$')
        plt.plot(x_params, reliability, marker='o', label=r'$\mathcal{M}_\text{R}$')
        plt.plot(x_params, compound, marker='o', label=r'$\mathcal{M}_\text{C}$')
        plt.axvline(x=default_params[n], label='Default', color='red', alpha=0.3, linestyle='--', linewidth=3)
        plt.xticks(x_params)
        plt.ylim(0.0, 1.1)
        plt.xlabel(f'{param_names_short[n]} {param_units[n]}')
        plt.ylabel('Performance [-]')
        plt.grid()
        plt.legend(loc='best', ncol=(len(performance_variables) + 1) / 4)
        plt.tight_layout()
        plt.savefig(f'figures/ConstellationComparison/sensitivity/all_{param_names[n]}_var_inc{default_params[0]}_T{default_params[1] * default_params[2]}_P{default_params[2]}_F{default_params[3]}_h{int(default_params[4] * 1e-3)}km_{int(default_params[5])}LCTs.png', dpi=125)
        # plt.show()

        # plot compound with discrete plus-minus labels
        default_index = np.where(x_params == default_x_param)[0][0]
        x_plusminus_single = []
        for j in range(len(x_params)):
            x_plusminus_single.append(j - default_index)
        x_plusminus = list(set(x_plusminus + x_plusminus_single))

        # append values to lists
        x_plusminus_collection.append(x_plusminus_single)
        compounds.append(compound)
        compound_names.append(param_names_short[n])

    plt.close('all') # close all previous plots

    plt.figure(figsize=(8, 7))
    # plt.title(f'{len(analysis_indices)} parameter variation \n'
    #           f'{default_params[0]}:{default_params[1] * default_params[2]}/{default_params[2]}/{default_params[3]} \n'
    #           f'at {int(default_params[4] * 1e-3)}km with {default_params[5]} LCTs')
    plt.title(f'{len(analysis_indices)} Parameter variation compound performance')
    for i in range(len(compounds)): # exclude default double
        plt.plot(x_plusminus_collection[i], compounds[i], marker='o', label=compound_names[i])
    plt.xticks(x_plusminus)
    plt.xlabel('Parameter index [-]')
    plt.ylabel('Performance [-]')
    plt.grid()
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(f'figures/ConstellationComparison/sensitivity/{save_indices}_param_var_inc{default_params[0]}_T{default_params[1] * default_params[2]}_P{default_params[2]}_F{default_params[3]}_h{int(default_params[4] * 1e-3)}km_{int(default_params[5])}LCTs.png', dpi=125)

    # save compound results
    compound_labels = ['min val', 'min idx', 'min decrease', 'max val', 'max idx', 'default increase']
    compound_dict = {
        'Labels': compound_labels,
        'Inclination': [],
        'Sats per plane': [],
        'Planes': [],
        'Phasing': [],
        'Altitude': [],
        'Terminals': []
    }

    print('----------')
    print('COMPOUND PERFORMANCE RESULTS:')
    print(f'Default compound = {default_compound}')
    for j in range(len(compounds)):
        # max values
        max_compound = np.max(compounds[j])
        max_idx = np.where(compounds[j] == max_compound)[0][0]
        max_plusminus_idx = x_plusminus_collection[j][max_idx]
        max_delta = (max_compound / default_compound - 1) * 100
        max_vals = [max_compound, max_plusminus_idx, max_delta]
        # min values
        min_compound = np.min(compounds[j])
        min_idx = np.where(compounds[j] == min_compound)[0][0]
        min_plusminus_idx = x_plusminus_collection[j][min_idx]
        min_delta = -(1 - min_compound / default_compound) * 100
        min_vals = [min_compound, min_plusminus_idx, min_delta]
        # append to dict
        min_max_total = min_vals + max_vals
        compound_dict[param_names[j]] = min_max_total

    for key in compound_dict:
        print(f"{key}: {compound_dict[key]}")

    plt.show()

if analyze_custom_constellation_collection:
    # set font size
    plt.rcParams.update({'font.size': 17})

    # all constellations
    constellation_names = [
        'Starlink PIV3 4',      # 0
        'Starlink PIV3 3',      # 1
        'IRIS2 LEO-H 4',        # 2
        'IRIS2 LEO-H 3',        # 3
        'Iridium 4',            # 4
        'Iridium 3',            # 5
        'OneWeb 4',             # 6
        'OneWeb 3',             # 7
        'NeLS 4',               # 8
        'NeLS 3',               # 9
        'SDA TT1 4',            # 10
        'SDA TT1 3',            # 11
        'Custom 4',              # 12
        'Custom 3'              # 13
    ]

    # 4 and 3 lct configurations
    constellation_set = [
        np.array([53, 72, 22, 17, 550.0E3, 4]),     # 0
        np.array([53, 72, 22, 17, 550.0E3, 3]),     # 1
        np.array([55, 12, 22, 10, 1200.0E3, 4]),    # 2
        np.array([55, 12, 22, 10, 1200.0E3, 3]),    # 3
        np.array([86.4, 11, 6, 1, 780.0E3, 4]),     # 4
        np.array([86.4, 11, 6, 1, 780.0E3, 3]),     # 5
        np.array([88, 40, 18, 1, 1200.0E3, 4]),     # 6
        np.array([88, 40, 18, 1, 1200.0E3, 3]),     # 7
        np.array([55, 12, 10, 1, 1200.0E3, 4]),     # 8
        np.array([55, 12, 10, 1, 1200.0E3, 3]),     # 9
        np.array([81, 21, 6, 1, 1000.0E3, 4]),      # 10
        np.array([81, 21, 6, 1, 1000.0E3, 3]),      # 11
        np.array([65, 10, 10, 5, 1200.0E3, 4]),     # 12
        np.array([65, 10, 10, 5, 1200.0E3, 3])      # 13
    ]

    # analysis_set = [4, 5, 8, 9, 10, 11]
    # analysis_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # analysis_set = [4, 5]
    # analysis_set = [2, 3, 4, 5]
    analysis_set = [4, 5, 8, 9, 10, 11, 12, 13]

    # simulation time
    t_frac = 1.0
    # t_frac = 0.1

    # number of routes
    number_routes = 8
    # number_routes = 3

    files = []
    efficiency = []
    users = []
    latency = []
    slew = []
    reliability = []
    compound = []
    for m in analysis_set:
        print('----------------')
        print(f'Constellation {constellation_names[m]} configuration is [i, S, P, F, h, LCT]:')
        print(constellation_set[m])

        # set the constellation configurations
        inclination = constellation_set[m][0]
        sats_total = constellation_set[m][1] * constellation_set[m][2]
        planes = constellation_set[m][2]
        sats_per_plane = constellation_set[m][1]
        phasing = constellation_set[m][3]
        altitude = constellation_set[m][4]
        lcts = constellation_set[m][5]

        # run the constellation mission file to simulate topology, routing and OISL's
        result, link_result, path_result = constellation_mission(
            name=constellation_names[m],
            inclination=inclination,
            sats_total=int(sats_total),
            planes=int(planes),
            sats_per_plane=int(sats_per_plane),
            phasing=int(phasing),
            altitude=altitude,
            lcts=int(lcts),
            number_routes=number_routes,
            t_frac=t_frac
        )

        files.append(result['transferred files'])
        efficiency.append(result['path efficiency'])
        users.append(result['user ratio'])
        latency.append(result['path latency'])
        slew.append(result['slew rate'])
        reliability.append(result['reliability'])
        compound.append(result['compound'])

        print('--------')
        print('Results:')
        print(result)

    # performance_labels = ['Files', 'Efficiency', 'Users', 'Latency', 'Slew', 'Reliability', 'Compound']
    performance_labels = [r'$\mathcal{M}_\text{D}$', r'$\mathcal{M}_\eta$', r'$\mathcal{M}_\text{U}$', r'$\mathcal{M}_\tau$', r'$\mathcal{M}_\omega$', r'$\mathcal{M}_\text{R}$', r'$\mathcal{M}_\text{C}$']

    # Bar settings
    bar_width = 0.08 # 0.05
    group_spacing = 0.4  # space between label groups
    font = 6

    # Calculate x positions
    x = np.arange(len(performance_labels)) * (len(constellation_set) * bar_width + group_spacing)

    plt.figure(figsize=(17, 7))
    plt.title(f'{int(len(analysis_set) / 2)} Constellation performance comparison')

    # bar plot
    for n in range(len(analysis_set)):
        shift = -0.5 * len(analysis_set) * bar_width + (n + 0.5) * bar_width
        # plot bars
        plt.bar(0 + shift, files[n], width=bar_width, color=color_list[n], edgecolor='k')
        plt.bar(1 + shift, efficiency[n], width=bar_width, color=color_list[n], edgecolor='k')
        plt.bar(2 + shift, users[n], width=bar_width, color=color_list[n], edgecolor='k')
        plt.bar(3 + shift, latency[n], width=bar_width, color=color_list[n], edgecolor='k')
        plt.bar(4 + shift, slew[n], width=bar_width, color=color_list[n], edgecolor='k')
        plt.bar(5 + shift, reliability[n], width=bar_width, color=color_list[n], edgecolor='k')
        plt.bar(6 + shift, compound[n], width=bar_width, label=constellation_names[analysis_set[n]], color=color_list[n], edgecolor='k')
        # plot values on top of bars
        # plt.text(0 + shift, files[n], f'{files[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
        # plt.text(1 + shift, efficiency[n], f'{efficiency[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
        # plt.text(2 + shift, users[n], f'{users[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
        # plt.text(3 + shift, latency[n], f'{latency[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
        # plt.text(4 + shift, slew[n], f'{slew[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
        # plt.text(5 + shift, reliability[n], f'{reliability[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
        # plt.text(6 + shift, compound[n], f'{compound[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
    plt.xticks(range(7), performance_labels, rotation=0) # 45 rotation
    # plt.xlabel('Parameters')
    plt.ylabel('Performance [-]')
    plt.legend(loc='lower left', ncol=4)
    plt.tight_layout()
    save_name = [constellation_names[analysis_set[0]], constellation_names[analysis_set[-1]]] # save according to first and last analysis index
    save_name = '_'.join(save_name)
    plt.savefig(f'figures/ConstellationComparison/Direct/Comparison_{save_name}.png', dpi=125)

    # difference bar plot
    files_dif = np.array(files)[::2] - np.array(files)[1::2]
    efficiency_dif = np.array(efficiency)[::2] - np.array(efficiency)[1::2]
    users_dif = np.array(users)[::2] - np.array(users)[1::2]
    latency_dif = np.array(latency)[::2] - np.array(latency)[1::2]
    slew_dif = np.array(slew)[::2] - np.array(slew)[1::2]
    reliability_dif = np.array(reliability)[::2] - np.array(reliability)[1::2]
    compound_dif = np.array(compound)[::2] - np.array(compound)[1::2]

    files_rel = (np.array(files)[::2] / np.array(files)[1::2] - 1.0) * 100.0
    efficiency_rel = (np.array(efficiency)[::2] / np.array(efficiency)[1::2] - 1.0) * 100.0
    users_rel = (np.array(users)[::2] / np.array(users)[1::2] - 1.0) * 100.0
    latency_rel = (np.array(latency)[::2] / np.array(latency)[1::2] - 1.0) * 100.0
    slew_rel = (np.array(slew)[::2] / np.array(slew)[1::2] - 1.0) * 100.0
    reliability_rel = (np.array(reliability)[::2] / np.array(reliability)[1::2] - 1.0) * 100.0
    compound_rel = (np.array(compound)[::2] / np.array(compound)[1::2] - 1.0) * 100

    # print difference results
    print('----------')
    print('4-3 difference results')
    print('Files:', files_dif, files_rel)
    print('Efficiency:', efficiency_dif, efficiency_rel)
    print('Users:', users_dif, users_rel)
    print('latency:', latency_dif, latency_rel)
    print('Slew:', slew_dif, slew_rel)
    print('Reliability:', reliability_dif, reliability_rel)
    print('Compound:', compound_dif, compound_rel)

    # Bar settings
    bar_width = 0.15  # 0.05
    group_spacing = 0.4  # space between label groups
    font = 8

    plt.figure(figsize=(17, 7)) # 12, 4
    plt.title(f'Constellation comparison LCT difference 4-3')
    for n in range(int(len(analysis_set) / 2)):
    # for n in range(1, int(len(analysis_set)), 2):
        # plot bars
        shift = -0.5 * (len(analysis_set) / 2) * bar_width + (n + 0.5) * bar_width
        plt.bar(0 + shift, files_dif[n], width=bar_width, color=color_list[n])
        plt.bar(1 + shift, efficiency_dif[n], width=bar_width, color=color_list[n])
        plt.bar(2 + shift, users_dif[n], width=bar_width, color=color_list[n])
        plt.bar(3 + shift, latency_dif[n], width=bar_width, color=color_list[n])
        plt.bar(4 + shift, slew_dif[n], width=bar_width, color=color_list[n])
        plt.bar(5 + shift, reliability_dif[n], width=bar_width, color=color_list[n])
        plt.bar(6 + shift, compound_dif[n], width=bar_width, label=constellation_names[analysis_set[2 * n]][:-2], color=color_list[n])
        # plot values on top of bars
        plt.text(0 + shift, files_dif[n], f'{files_dif[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
        plt.text(1 + shift, efficiency_dif[n], f'{efficiency_dif[n]:.2f}', ha='center', va='bottom', fontsize=font,color=color_list[n])
        plt.text(2 + shift, users_dif[n], f'{users_dif[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
        plt.text(3 + shift, latency_dif[n], f'{latency_dif[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
        plt.text(4 + shift, slew_dif[n], f'{slew_dif[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
        plt.text(5 + shift, reliability_dif[n], f'{reliability_dif[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
        plt.text(6 + shift, compound_dif[n], f'{compound_dif[n]:.2f}', ha='center', va='bottom', fontsize=font, color=color_list[n])
    plt.axhline(y=0, color='k')
    # plt.yscale('log')
    plt.yscale('symlog', linthresh=1)
    plt.xticks(range(7), performance_labels, rotation=45)
    # plt.xlabel('Parameters')
    plt.ylabel('Performance [-]')
    plt.legend(loc='upper right', ncol=2)
    plt.tight_layout()
    save_name = [constellation_names[analysis_set[0]], constellation_names[analysis_set[-1]]] # save according to first and last analysis index
    save_name = '_'.join(save_name)
    plt.savefig(f'figures/ConstellationComparison/Direct/LCTComparison_{save_name}.png', dpi=125)

    plt.show()


if analyze_latency_literature:
    ############# LITERATURE STUDY COMPARISON ################
    # PAPER: Temporary Laser Inter-Satellite Links in Free-Space Optical Satellite Networks
    # by Chaudry and yanikomeroglu
    ##########################################################
    # The path latency from the paper will be compared by the
    # values produced by the model for the routes in the paper
    ##########################################################

    constellation_name = 'Starlink P1V2'
    constellation_set = [53, 66, 24, 0, 550.0E3, 4] # 0 phasing (uniform distribution assumed in paper), 4 lct
    delay_node = 10.0E-3 # 10 ms
    range_links = np.array([1500.0E3, 1700.0E3])

    # simulation time
    t_frac = 1.0
    link_step = 1.0 # s

    # number of routes
    number_routes = 4

    print('----------------')
    print(f'Constellation {constellation_name} configuration is [i, S, P, F, h, LCT]:')
    print(constellation_set)
    print(f'Reference node delay = {delay_node} s and link ranges = {range_links * 1e-3} km')

    # set the constellation configurations
    inclination = constellation_set[0]
    sats_total = constellation_set[1] * constellation_set[2]
    planes = constellation_set[2]
    sats_per_plane = constellation_set[1]
    phasing = constellation_set[3]
    altitude = constellation_set[4]
    lcts = constellation_set[5]

    # run the constellation mission file to simulate topology, routing and OISL's
    result, link_result, path_result = constellation_mission(
        name=constellation_name,
        inclination=inclination,
        sats_total=int(sats_total),
        planes=int(planes),
        sats_per_plane=int(sats_per_plane),
        phasing=int(phasing),
        altitude=altitude,
        lcts=int(lcts),
        t_frac=t_frac,
        link_step=link_step,
        number_routes=number_routes,
        link_simulation=False,
        shared_links=False,
        RF_latency=True
    )

    files = result['transferred files']
    efficiency = result['path efficiency']
    users = result['user ratio']
    latency = result['path latency']
    slew = result['slew rate']
    reliability = result['reliability']
    compound = result['compound']

    # append the latencies
    latencies_min = np.array([np.min(path_result[n]['path latency']) for n in range(len(path_result))])
    latencies_mean = np.array([np.mean(path_result[n]['path latency']) for n in range(len(path_result))])
    latencies_max = np.array([np.max(path_result[n]['path latency']) for n in range(len(path_result))])
    latencies_std = np.array([np.std(path_result[n]['path latency']) for n in range(len(path_result))])
    latencies_rms = np.array([np.sqrt(np.mean((path_result[n]['path latency'] - np.mean(path_result[n]['path latency']))**2)) for n in range(len(path_result))])

    print('--------')
    print('Results:')
    print(result)

    # latency study information
    performance_labels = [
        ['Sydney', 'Sao Paolo'],
        ['Toronto', 'Istanbul'],
        ['Madrid', 'Tokyo'],
        ['New York', 'Jakarta']
    ]

    latency_study_combinations = [
        [performance_labels[0], range_links[0]],
        [performance_labels[0], range_links[1]],
        [performance_labels[1], range_links[1]],
        [performance_labels[2], range_links[1]],
        [performance_labels[3], range_links[1]]
    ]

    hops_study_results = [
        13.5,
        13,
        7.5,
        10,
        15
    ]

    latency_study_results = [
        190 - delay_node * 1e3 * hops_study_results[0],
        181 - delay_node * 1e3 * hops_study_results[1],
        110 - delay_node * 1e3 * hops_study_results[2],
        149 - delay_node * 1e3 * hops_study_results[3],
        215 - delay_node * 1e3 * hops_study_results[4]
    ]

    latency_rms_study_0 = np.array([np.sqrt(np.mean((path_result[0]['path latency'] - latency_study_results[0] * 1e-3) ** 2))])
    latencies_rms_study_1_4 = np.array([np.sqrt(np.mean((path_result[n]['path latency'] - latency_study_results[n+1] * 1e-3) ** 2)) for n in range(len(path_result))])
    latencies_rms_study = np.concatenate((latency_rms_study_0, latencies_rms_study_1_4))

    # Bar settings
    bar_width = 0.3
    group_spacing = 0.05  # space between label groups

    # Calculate x positions
    x = np.arange(len(performance_labels)) * (len(range_links) * bar_width + group_spacing)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.title(f'Latency StarLink P1V2 ({int(inclination)}:{int(sats_total)}/{int(planes)}/{int(phasing)} at {int(altitude * 1e-3)} km, time = {int(3600 * t_frac)} s)')

    for n in range(len(latencies_mean)):
        bar_height = latencies_mean[n] * 1e3

        # with min, max values
        # bar_lower = latencies_min[n] * 1e3
        # bar_upper = latencies_max[n] * 1e3
        # min_err = bar_height - bar_lower
        # max_err = bar_upper - bar_height
        # errs = [[min_err], [max_err]]

        # with standard deviation values
        std_factor = 1.0
        bar_lower = bar_height - latencies_std[n] * 1e3 * std_factor
        bar_upper = bar_height + latencies_std[n] * 1e3 * std_factor
        min_err = latencies_std[n] * 1e3 * std_factor
        max_err = latencies_std[n] * 1e3 * std_factor
        errs = [[min_err],[max_err]]
        # print(errs)

        # with RMS values
        # bar_lower = bar_height - latencies_rms[n] * 1e3
        # bar_upper = bar_height + latencies_rms[n] * 1e3
        # min_err = latencies_rms[n] * 1e3
        # max_err = latencies_rms[n] * 1e3
        # errs = [[min_err],[max_err]]
        # print(errs)

        # with RMS study values
        # bar_lower = bar_height - latencies_rms_study[n] * 1e3
        # bar_upper = bar_height + latencies_rms_study[n] * 1e3
        # min_err = latencies_rms_study[n] * 1e3
        # max_err = latencies_rms_study[n] * 1e3
        # errs = [[min_err],[max_err]]
        # print(errs)

        if n == 0:
            rRMSE = np.round([latencies_rms_study[0] / (latency_study_results[0] * 1e-3) * 100, latencies_rms_study[1] / (latency_study_results[1] * 1e-3) * 100], 1)
            error_label = fr'$\sigma$ = {np.round(max_err, 1)} ms ' + '($rRMSE_{1500,1700} = $' + f'{rRMSE[0]} %, {rRMSE[1]} %)'
        else:
            rRMSE = np.round(latencies_rms_study[n+1] * 1e3 / latency_study_results[n+1] * 100, 1)
            error_label = fr'$\sigma$ = {np.round(max_err, 1)} ms ' + '($rRMSE_{1700} = $' + f'{rRMSE} %)'

        ax.bar(n, bar_height, yerr=errs, capsize=15, width=bar_width, label=error_label, edgecolor='k', color=color_list[n], zorder=5)

    for m in range(len(latency_study_results)):
        if m == 0 or m == 1:
            current_color = color_list[0]
        else:
            current_color = color_list[m-1]
        ax.axhline(y=latency_study_results[m], color=current_color, linestyle='--', zorder=0)
        label = str(latency_study_combinations[m][0]) + ' ' + str(int(latency_study_combinations[m][1] * 1e-3)) + ' km ' + str(latency_study_results[m]) + ' ms'
        ax.text(1.0, latency_study_results[m], label, bbox=dict(facecolor='white', edgecolor='black'),ha='left', va='center', transform=ax.get_yaxis_transform())

    # # bar plot
    # for n in range(len(range_links)):
    #     shift = -0.5 * len(range_links) * bar_width + (n + 0.5) * bar_width
    #     for m in range(len(latencies_mean[n])):
    #         combo = [performance_labels[m], range_links[n]]
    #         if combo in latency_study_combinations:
    #             bar_height = latencies_mean[n][m] * 1e3
    #             min_err = bar_height - (latencies_min[n][m] * 1e3)
    #             max_err = (latencies_max[n][m] * 1e3) - bar_height
    #             errs = [[min_err], [max_err]]
    #             ax.bar(m + shift, bar_height, yerr=errs, capsize=7, width=bar_width, edgecolor='k', label=f'{int(range_links[n] * 1e-3)} simulation', color=color_list[n], zorder=5)
    #
    #             study_index = latency_study_combinations.index(combo)
    #             label = str(latency_study_combinations[study_index][0]) + ' ' + str(int(latency_study_combinations[study_index][1] * 1e-3)) + ' km ' + str(latency_study_results[study_index]) + ' ms'
    #             ax.axhline(y=latency_study_results[study_index], color=color_list[n], linestyle='--', zorder=0)
    #             ax.text(1.0, latency_study_results[study_index], label, bbox=dict(facecolor='white', edgecolor='black'), ha='left', va='center', transform=ax.get_yaxis_transform())

    plt.xticks(range(len(performance_labels)), performance_labels, rotation=45)
    plt.ylabel('Mean latency [ms]')

    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), loc='lower left')

    ax.legend(loc='lower left')

    plt.tight_layout()
    save_name = constellation_name.replace(' ', '_')
    plt.savefig(f'figures/ConstellationComparison/Latency_study_{save_name}.png')

    plt.show()
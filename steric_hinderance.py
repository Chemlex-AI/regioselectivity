import imp
from rdkit import Chem
from rdkit.Chem import AllChem
import miniball
import numpy as np
import pandas as pd
import math
import temp_ext as te
import re
from collections import Counter
import temp_ext as te
from collections import defaultdict
import random

numConfs = 20
thread_per_conf = 0   # 0 == no limit
seed = 123

def read_one_xyz(filename):
    with open(filename, 'r') as f:
        content = f.read()
        contact = content.split('\n')
        for line in contact:
            if line == '' or line.isdigit():
                continue
            elif '\t' in line or '\n' in line:
                atom = line.split()
    return atom

def remove_smi_am(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    smi = Chem.MolToSmiles(mol)
    return smi


def remove_smr_am(smr):
    mol = Chem.MolFromSmarts(smr)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    smr = Chem.MolToSmarts(mol)
    return smr


def atom_num_to_xyz(alist, df):
    anarray = np.zeros((len(alist), 3))
    for i, atomnum in enumerate(alist):
        anarray[i] = np.array(df.iloc[atomnum][['x', 'y', 'z']], dtype=np.float64)
    return anarray


def find_center(match_array):
    C, r2 = miniball.get_bounding_ball(match_array)
    return C, (r2) ** (0.5)


def cart2sph(*cord):
    try:
        x, y, z = cord
    except:
        x, y, z = cord[0]
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
    theta = math.acos(z/r)
    phi = math.atan2(y, x)
    return r, theta, phi


def sph2cart(*cord):
    try:
        r, theta, phi = cord
    except:
        r, theta, phi = cord[0]
    x = r * math.sin( theta ) * math.cos( phi )
    y = r * math.sin( theta ) * math.sin( phi )
    z = r * math.cos( theta )
    return x, y, z


def cart2sph_array(array):
    return np.array([cart2sph(item) for item in array])


def sph2cart_array(array):
    return np.array([sph2cart(item) for item in array])


def eucliDist(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))


def vdw_r(atom):
    return Chem.PeriodicTable.GetRvdw(Chem.GetPeriodicTable(), atom)

#deprived 
# def cal_intersect(r, R, d):
#     if d <= R - r:
#         return 4 / 3 * math.pi * (math.pow(r, 3))
#     elif d <= R:
#         h1 = R - ((math.pow(R, 2) + math.pow(d, 2) - math.pow(r, 2)) / (2 * d))
#         h2 = r - ((math.pow(r, 2) + math.pow(d, 2) - math.pow(R, 2)) / (2 * d))
#         v1 = math.pi * math.pow(h1, 2) * (R - h1 / 3)
#         v2 = (4 / 3 * math.pi * (math.pow(r, 3))) - (math.pi * math.pow(h2, 2) * (r - h2 / 3))
#         return v1 + v2
#     elif d < r + R:
#         h1 = R - ((math.pow(R, 2) + math.pow(d, 2) - math.pow(r, 2)) / (2 * d))
#         h2 = r - ((math.pow(r, 2) + math.pow(d, 2) - math.pow(R, 2)) / (2 * d))
#         v1 = math.pi * math.pow(h1, 2) * (R - h1 / 3)
#         v2 = math.pi * math.pow(h2, 2) * (r - h2 / 3)
#         return v1 + v2
#     else:
#         return 0


def get_randm_points(center, radius, n):
    rng = np.random.default_rng(seed=seed)
    r1, r2 = 0, radius
    theta1, theta2 = 0, math.pi
    phi1, phi2 = 0, 2 * math.pi
    n = n
    samples = rng.uniform([r1, theta1, phi1], [r2, theta2, phi2], size=(n, 3))
    samples = sph2cart_array(samples)
    samples = samples + center
    return samples


def get_crit_perct(remainlist, matchlist, df, crit_rad, n):
    remain_xyz = atom_num_to_xyz(remainlist, df)
    match_xyz = atom_num_to_xyz(matchlist, df)

    center, _ = find_center(match_xyz)
    rads = [vdw_r(df.iloc[num]['symbol']) for num in remainlist]
    samples = get_randm_points(center, crit_rad, n)
    ct = 0

    for i in range(n):
        for j in range(len(remain_xyz)):
            if eucliDist(samples[i], remain_xyz[j]) < rads[j]:
                ct +=1
                break
    return ct / n



# def get_crit_perct(remainlist, matchlist, df, crit_rad):
#     remain_xyz = atom_num_to_xyz(remainlist, df)
#     match_xyz = atom_num_to_xyz(matchlist, df)

#     center, _ = find_center(match_xyz)
#     dist = [eucliDist(center, array) for array in remain_xyz]
#     rads = [vdw_r(df.iloc[num]['symbol']) for num in remainlist]
#     crit_dist = np.array(dist) - np.array(rads)
#     # crit_atoms_array = remain_array[np.argwhere(crit_dist < crit_rad)]
#     # crit_atoms_list = np.array(remainlist)[np.argwhere(crit_dist < crit_rad)].reshape(-1).tolist()

#     asso_atom_r = np.array(rads)[np.argwhere(crit_dist < crit_rad)].reshape(-1)
#     asso_atom_d = np.array(dist)[np.argwhere(crit_dist < crit_rad)].reshape(-1)
#     # asso_atom_d_perct = np.array(
#     #     [min((crit_rad - asso_atom_d)[i], (2 * asso_atom_r)[i]) for i in range(len(crit_rad - asso_atom_d))])
#     #
#     # asso_atom_d_perct_sum = sum(asso_atom_d_perct) / crit_rad

#     assert len(asso_atom_r) == len(asso_atom_d)
#     asso_atom_v = []
#     for i in range(len(asso_atom_d)):
#         v = cal_intersect(asso_atom_r[i], crit_rad, asso_atom_d[i])
#         asso_atom_v.append(v)
#     asso_atom_v_array = np.array(asso_atom_v)
#     V = 4 / 3 * math.pi * math.pow(crit_rad, 3)
#     return asso_atom_v_array.sum() / V


def mol_conf(target_smi, xyz_path='', confmollst=''):
    if confmollst:
        target_mol = Chem.MolFromSmiles(target_smi)

        all_atom_list = Chem.MolToMolBlock(target_mol).split('\n')[4:]  # remove heading
        atom_list = []

        for item in all_atom_list:
            temp = item.split()
            if len(temp) > 15:
                atom_list.append([temp[i] for i in [0, 1, 2, 3, -3]])
            if len(temp) == 15:
                atom_list.append([temp[i] for i in [0, 1, 2, 3]] + [temp[-3][1:]])

        col = ['x', 'y', 'z', 'symbol', 'atommappingnum']
        df = pd.DataFrame(data=atom_list, columns=col)
        dfdic = {}
        if len(confmollst) >= 3:
            rmsd_list = []
            for i in range(len(confmollst)):
                alst = []
                for j in range(len(confmollst)):
                    try:
                        rmsd_val = AllChem.GetBestRMS(Chem.RemoveHs(confmollst[i]), Chem.RemoveHs(confmollst[j]))
                    except RuntimeError:
                        return None
                    alst.append(rmsd_val)
                rmsd_list.append(alst)

            total_dis_list = []
            for a in range(len(confmollst)):
                for b in range(len(confmollst)):
                    for c in range(len(confmollst)):
                        total_dis_list.append(
                            math.pow((rmsd_list[a][b] ** 2 + rmsd_list[b][c] ** 2 + rmsd_list[c][a] ** 2), 0.5))

            total_dis_array = np.array(total_dis_list)
            t = np.where(total_dis_array == total_dis_array.max())[0][0]
            c = t % len(confmollst)
            b = t // len(confmollst) % len(confmollst)
            a = t // (len(confmollst) * len(confmollst)) % len(confmollst)
            confa = confmollst[a]
            confb = confmollst[b]
            confc = confmollst[c]
            dfdic = {}
            for i in range(3):
                dfdic[i] = pd.DataFrame()

            dfdic[0][['x', 'y', 'z']] = confa.GetConformer(id=0).GetPositions()
            dfdic[1][['x', 'y', 'z']] = confb.GetConformer(id=0).GetPositions()
            dfdic[2][['x', 'y', 'z']] = confc.GetConformer(id=0).GetPositions()

        if len(confmollst) == 2:
            for i in range(2):
                dfdic[i] = pd.DataFrame()
            dfdic[0][['x', 'y', 'z']] = confmollst[0].GetConformer(id=0).GetPositions()
            dfdic[1][['x', 'y', 'z']] = confmollst[1].GetConformer(id=0).GetPositions()
        if len(confmollst) == 1:
            dfdic[0] = pd.DataFrame()
            dfdic[0][['x', 'y', 'z']] = confmollst[0].GetConformer(id=0).GetPositions()

        for i in range(len(dfdic)):
            dfdic[i][['symbol', 'atommappingnum']] = df[['symbol', 'atommappingnum']]
        idx2am_map = {}
        for num in range(len(atom_list)):
            idx2am_map[num] = atom_list[num][-1]
        return dfdic, idx2am_map

    if xyz_path:
        target_mol = Chem.MolFromSmiles(target_smi)

        all_atom_list = Chem.MolToMolBlock(target_mol).split('\n')[4:]  # remove heading
        atom_list = []

        for item in all_atom_list:
            temp = item.split()
            if len(temp) > 15:
                atom_list.append([temp[i] for i in [0, 1, 2, 3, -3]])
            if len(temp) == 15:
                atom_list.append([temp[i] for i in [0, 1, 2, 3]] + [temp[-3][1:]])

        col = ['x', 'y', 'z', 'symbol', 'atommappingnum']
        df = pd.DataFrame(data=atom_list, columns=col)
        df[['symbol', 'x', 'y', 'z']] = read_one_xyz(xyz_path)
        idx2am_map = {}
        for num in range(len(atom_list)):
            idx2am_map[num] = atom_list[num][-1]
        return df, idx2am_map

    else:
        target_mol = Chem.MolFromSmiles(target_smi)
        target_mol = Chem.AddHs(target_mol)
        cids = AllChem.EmbedMultipleConfs(target_mol, numConfs=numConfs, randomSeed=123, numThreads=thread_per_conf)
        target_mol = Chem.RemoveHs(target_mol)

        all_atom_list = Chem.MolToMolBlock(target_mol).split('\n')[4:]  # remove heading
        atom_list = []

        for item in all_atom_list:
            temp = item.split()
            if len(temp) > 15:
                atom_list.append([temp[i] for i in [0, 1, 2, 3, -3]])
            if len(temp) == 15:
                atom_list.append([temp[i] for i in [0, 1, 2, 3]] + [temp[-3][1:]])

        col = ['x', 'y', 'z', 'symbol', 'atommappingnum']
        df = pd.DataFrame(data=atom_list, columns=col)
        dis_list = []
        dfdic = {}
        denominator = min(len(cids), numConfs)
        if denominator >= 3:
            for i in range(denominator):
                blist = []
                for j in range(denominator):
                    blist.append(AllChem.GetConformerRMS(target_mol, i, j, prealigned=True))
                dis_list.append(blist)

            total_dis_list = []
            for a in range(denominator):
                for b in range(denominator):
                    for c in range(denominator):
                        total_dis_list.append(math.pow((dis_list[a][b] ** 2 + dis_list[b][c] ** 2 + dis_list[c][a] ** 2), 0.5))

            total_dis_array = np.array(total_dis_list)
            t = np.where(total_dis_array == total_dis_array.max())[0][0]
            c = t % denominator
            b = t // denominator % denominator
            a = t // (denominator * denominator) % denominator
            confa = target_mol.GetConformer(id=int(a))
            confb = target_mol.GetConformer(id=int(b))
            confc = target_mol.GetConformer(id=int(c))
            for i in range(3):
                dfdic[i] = pd.DataFrame()
            
            dfdic[0]['x'] = confa.GetPositions()[:,0]
            dfdic[0]['y'] = confa.GetPositions()[:,1]
            dfdic[0]['z'] = confa.GetPositions()[:,2]
            dfdic[1]['x'] = confb.GetPositions()[:,0]
            dfdic[1]['y'] = confb.GetPositions()[:,1]
            dfdic[1]['z'] = confb.GetPositions()[:,2]
            dfdic[2]['x'] = confc.GetPositions()[:,0]
            dfdic[2]['y'] = confc.GetPositions()[:,1]
            dfdic[2]['z'] = confc.GetPositions()[:,2]
        if denominator == 2:
            for i in range(2):
                dfdic[i] = pd.DataFrame()
            dfdic[0]['x'] = target_mol.GetConformer(id=0).GetPositions()[:,0]
            dfdic[0]['y'] = target_mol.GetConformer(id=0).GetPositions()[:,1]
            dfdic[0]['z'] = target_mol.GetConformer(id=0).GetPositions()[:,2]
            dfdic[1]['x'] = target_mol.GetConformer(id=1).GetPositions()[:,0]
            dfdic[1]['y'] = target_mol.GetConformer(id=1).GetPositions()[:,1]
            dfdic[1]['z'] = target_mol.GetConformer(id=1).GetPositions()[:,2]
    
        if denominator == 1:
            dfdic[0] = pd.DataFrame()
            dfdic[0]['x'] = target_mol.GetConformer(id=0).GetPositions()[:,0]
            dfdic[0]['y'] = target_mol.GetConformer(id=0).GetPositions()[:,1]
            dfdic[0]['z'] = target_mol.GetConformer(id=0).GetPositions()[:,2]

        for i in range(len(dfdic)):
            dfdic[i][['symbol', 'atommappingnum']] = df[['symbol', 'atommappingnum']]
        idx2am_map = {}
        for num in range(len(atom_list)):
            idx2am_map[num] = atom_list[num][-1]
        return dfdic, idx2am_map


def mol_match(target_smi, sub_smr, idx2am_map, consider_am=True):
    target_mol = Chem.MolFromSmiles(target_smi)
    sub_mol = Chem.MolFromSmarts(sub_smr)
    matchlists = list(target_mol.GetSubstructMatches(sub_mol))
    remainlists = [list(set([num for num in range(target_mol.GetNumAtoms())]) - set(matchlist)) for matchlist in matchlists]

    if consider_am:
        sub_smr_label = set(re.findall('\:([0-9]+)\]', sub_smr))
        for i in range(len(matchlists)):
            atom_map_num_list = [idx2am_map[idx] for idx in matchlists[i]]
            if not (set(atom_map_num_list) - sub_smr_label) or not (sub_smr_label - set(atom_map_num_list)):
                temp_matchlist = list(matchlists[i])
                temp_remainlist = remainlists[i]
        return temp_matchlist, temp_remainlist

    if not consider_am:
        return matchlists, remainlists


def sh_value(mapped_rxn, crit_rad, xyz_path='', confmollst='', n=None):

    reaction = {
        'reactants': mapped_rxn.split('>>')[0],
        'products': mapped_rxn.split('>>')[1],
        '_id': 1
    }
    template = te.extract_from_reaction(reaction, keep_atom_mapping=True, r_radius=1, p_radius=0)['reaction_smarts']
    target_smi = mapped_rxn.split('>>')[1]
    p_sub_smr, _ = template.split('>>')
    if not xyz_path:
        dfdic, idx2am_map = mol_conf(target_smi=target_smi, confmollst=confmollst)
        matchlist, remainlist = mol_match(target_smi, p_sub_smr, idx2am_map)
        cv = np.zeros(len(dfdic))
        for i in range(len(dfdic)):
            cv[i] = get_crit_perct(remainlist, matchlist, dfdic[i], crit_rad, n)
        return cv.mean()
    if xyz_path:
        df, idx2am_map = mol_conf(target_smi=target_smi, xyz_path=xyz_path)
        matchlist, remainlist = mol_match(target_smi, p_sub_smr, idx2am_map)
        cv = get_crit_perct(remainlist, matchlist, df, crit_rad, n)
        return cv


def sh_value_sample(smiles, sub_smr, crit_rad, xyz_path='', confmollst='', n=None):
    if not xyz_path:
        dfdic, idx2am_map = mol_conf(target_smi=smiles, confmollst=confmollst)
        matchlists, remainlists = mol_match(smiles, sub_smr, idx2am_map, consider_am=False)

        ccv = np.zeros(len(matchlists))
        for i in range(len(matchlists)):
            cv = np.zeros(len(dfdic))
            for j in range(len(dfdic)):
                cv[j] = get_crit_perct(remainlists[i], matchlists[i], dfdic[j], crit_rad, n)
            ccv[i] = cv.mean()
        return ccv
    if xyz_path:
        df, idx2am_map = mol_conf(target_smi=smiles, xyz_path=xyz_path)
        matchlists, remainlists = mol_match(smiles, sub_smr, idx2am_map, consider_am=False)

        ccv = np.zeros(len(matchlists))
        for i in range(len(matchlists)):
            cv = get_crit_perct(remainlists[i], matchlists[i], df, crit_rad, n)
            ccv[i] = cv
        return ccv


def sh_value_reactants(mapped_rxn, crit_rad, xyz_path='', confmollst='', n=None):
    reaction = {
        'reactants': mapped_rxn.split('>>')[0],
        'products': mapped_rxn.split('>>')[1],
        '_id': 1
    }
    template = te.extract_from_reaction(reaction, keep_atom_mapping=True, r_radius=1, p_radius=0)['reaction_smarts']
    _, r_sub_smrs = template.split('>>')
    r_sub_smr_lst = r_sub_smrs.split('.')
    reactants_smis = mapped_rxn.split('>>')[0].split('.')
    label_to_r_smi = {}
    for r_smi in reactants_smis:
        r_smi_label = set(re.findall('\:([0-9]+)\]', r_smi))
        label_to_r_smi[str(r_smi_label)] = r_smi
    r_sub_smr_to_label = {}
    for r_sub_smr in r_sub_smr_lst:
        r_sub_smr_label = set(re.findall('\:([0-9]+)\]', r_sub_smr))
        r_sub_smr_to_label[r_sub_smr] = r_sub_smr_label

    r_sub_smr_used = []
    temp = []
    if not xyz_path:
        for r_sub_smr in r_sub_smr_lst:
            if r_sub_smr in r_sub_smr_used:
                continue
            for item in label_to_r_smi:
                if not (r_sub_smr_to_label[r_sub_smr] - eval(item)):
                    dfdic, idx2am_map = mol_conf(target_smi=label_to_r_smi[item], confmollst=confmollst)
                    matchlist, remainlist = mol_match(label_to_r_smi[item], r_sub_smr, idx2am_map)
                    temp.append((dfdic, matchlist, remainlist))
                    r_sub_smr_used.append(r_sub_smr)
                    break
        try:
            assert Counter(r_sub_smr_used) == Counter(r_sub_smr_lst)
        except Exception as e:
            return f'error: {str(e)}'

        ccv = np.zeros(len(temp))
        for j in range(len(temp)):
            dfdic, matchlist, remainlist = temp[j]
            cv = np.zeros(len(dfdic))
            for i in range(len(dfdic)):
                cv[i] = get_crit_perct(remainlist, matchlist, dfdic[i], crit_rad, n)
            ccv[j] = cv.mean()

        return ccv.sum()

    if xyz_path:
        for r_sub_smr in r_sub_smr_lst:
            if r_sub_smr in r_sub_smr_used:
                continue
            for item in label_to_r_smi:
                if not (r_sub_smr_to_label[r_sub_smr] - eval(item)):
                    df, idx2am_map = mol_conf(target_smi=label_to_r_smi[item], xyz_path=xyz_path)
                    matchlist, remainlist = mol_match(label_to_r_smi[item], r_sub_smr, idx2am_map)
                    temp.append((df, matchlist, remainlist))
                    r_sub_smr_used.append(r_sub_smr)
                    break
        try:
            assert Counter(r_sub_smr_used) == Counter(r_sub_smr_lst)
        except Exception as e:
            return f'error: {str(e)}'

        ccv = np.zeros(len(temp))
        for j in range(len(temp)):
            df, matchlist, remainlist = temp[j]
            cv = get_crit_perct(remainlist, matchlist, df, crit_rad, n)
            ccv[j] = cv
        return ccv.sum()



def sh_value_reactants_sample(mapped_rxn='',sub_smrs=[], crit_rad=None, xyz_path='', confmollst='', n=None):
    if isinstance(mapped_rxn, str):
        reactants_smis = mapped_rxn.split('>>')[0].split('.')
    if isinstance(mapped_rxn, list):
        reactants_smis = mapped_rxn
    reactants_mols = [Chem.MolFromSmiles(smi) for smi in reactants_smis]
    if isinstance(sub_smrs, str):
        sub_smrs = sub_smrs.split('.')
    if isinstance(sub_smrs, list):
        sub_smrs = sub_smrs
    sub_structs = [Chem.MolFromSmarts(smr) for smr in sub_smrs]

    asign_dict = defaultdict(list)
    for i in range(len(sub_structs)):
        for j in range(len(reactants_mols)):
            if reactants_mols[j].HasSubstructMatch(sub_structs[i]):
                asign_dict[sub_smrs[i]].append(reactants_smis[j])
    asign_dict = dict(asign_dict)

    found_asignment = False
    while not found_asignment:
        asignment = dict()
        reactants_used = []
        for smr in sub_smrs:
            idx = random.randint(0,len(asign_dict[smr])-1)
            if asign_dict[smr][idx] not in reactants_used:
                asignment[smr] = asign_dict[smr][idx]
                reactants_used.append(asign_dict[smr][idx])
            else:
                break
        if Counter(sub_smrs) == Counter(list(asignment.keys())):
            found_asignment = True

    ccv = 0
    for sub_smr in sub_smrs:
        ccv += sh_value_sample(smiles=asignment[sub_smr], sub_smr=sub_smr, crit_rad=crit_rad, xyz_path=xyz_path, confmollst=confmollst, n=n).min()
    return ccv


def master_sh(adict):
    mapped_rxn=adict['mapped_rxn']
    smiles=adict['smiles'] if 'smiles' in adict else ''
    sub_smr=adict['sub_smr'] if 'sub_smr' in adict else None
    sub_smrs=adict['sub_smrs'] if 'sub_smrs' in adict else []
    crit_rad=adict['crit_rad']
    xyz_path=adict['xyz_path']
    confmollst=adict['confmollst']
    n=adict['n']
    if adict['method'] == 'sh':
        return sh_value(mapped_rxn=mapped_rxn, crit_rad=crit_rad, xyz_path=xyz_path, confmollst=confmollst, n=n)
    if adict['method'] == 'shs':
        return sh_value_sample(smiles=smiles, sub_smr=sub_smr, crit_rad=crit_rad, xyz_path=xyz_path, confmollst=confmollst, n=n)
    if adict['method'] == 'shr':
        return sh_value_reactants(mapped_rxn=mapped_rxn, crit_rad=crit_rad, xyz_path=xyz_path, confmollst=confmollst, n=n)
    if adict['method'] == 'shrs':
        return sh_value_reactants_sample(mapped_rxn=mapped_rxn,sub_smrs=sub_smrs, crit_rad=crit_rad, xyz_path=xyz_path, confmollst=confmollst, n=n)
    
    


if __name__ == '__main__':
    pass

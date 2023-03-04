from dataclasses import dataclass


@dataclass
class Loss:
    loss_value = 0
    loss_rec_rna_v = 0
    loss_rec_msi_v = 0
    loss_trls_msi_v = 0
    loss_trls_rna_v = 0
    loss_rec_rna_poe_v = 0
    loss_rec_msi_poe_v = 0
    loss_kl_v = 0
    loss_kl1_p_v = 0
    loss_kl1_p_mod_v = 0
    loss_kl1_s_v = 0
    loss_kl2_p_v = 0
    loss_kl2_p_mod_v = 0
    loss_kl2_s_v = 0
    loss_mmd_v = 0
    loss_shared_v = 0
    loss_private_v = 0
    loss_private_mod_v = 0
    loss_batch_mod1_v = 0
    loss_batch_mod2_v = 0
    loss_cos1_v = 0
    loss_cos2_v = 0

    ####################################
    loss_batch_1_v = 0
    loss_batch_2_v = 0
    ####################################

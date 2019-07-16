import torch
from torch.distributions import Normal, Independent, kl

# Normals with batch_shape=(3,) and event_shape=()
dist1 = Normal(torch.zeros(4), torch.ones(4))
dist2 = Normal(torch.zeros(4), 2 * torch.ones(4))

# Normals with batch_shape=() and event_shape=(3,)
dist1_indep = Independent(dist1, 1)
dist2_indep = Independent(dist2, 1)

# Works fine, returns KL with shape (3,), one for each Normal
kl_divergence = kl.kl_divergence(dist1, dist2)

# Raises NotImplementedError, should return KL with shape (), the sum of the KL for each Normal
kl_divergence_indep = kl.kl_divergence(dist1_indep, dist2_indep)

print(f"kl divergence:{kl_divergence}")
print(f"kl divergence independent: {kl_divergence_indep}")

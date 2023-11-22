# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Documenter, KernelDistributions
import Documenter.Remotes: GitLab

makedocs(sitename="KernelDistributions", repo=GitLab("git-ce.rwth-aachen.de", "diss", "kerneldistributions.jl"))
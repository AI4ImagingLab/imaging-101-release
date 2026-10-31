"""Complete fragment rendering and common-C4 pose metrics for review figures.

Arrays are passed by caller; no reconstruction cache, paths or hidden assets are
loaded here. Importing this module never selects a Matplotlib backend.
"""
import numpy as np


def _pose(positions,quarter_turns):
    p=np.asarray(positions);q=np.asarray(quarter_turns)
    if p.ndim==3 and p.shape[0]==1:p=p[0]
    if q.ndim==2 and q.shape[0]==1:q=q[0]
    if p.ndim!=2 or p.shape[1]!=2 or q.shape!=(len(p),) or len(p)==0:
        raise ValueError('expected positions(N,2) and quarter_turns(N), optionally batch1')
    if p.dtype.kind not in 'iu' or q.dtype.kind not in 'iu':raise ValueError('integer poses required')
    if np.any(p<0) or np.any(q>3) or np.any(q<0):raise ValueError('invalid coordinate or quarter turn')
    if not np.array_equal(p.min(axis=0),[0,0]):raise ValueError('pose origin must be (0,0)')
    extent=p.max(axis=0).astype(np.int64)+1
    if int(np.prod(extent))!=len(p) or len(np.unique(p,axis=0))!=len(p):
        raise ValueError('complete collision-free rectangular pose required')
    return p.astype(np.int64,copy=False),q.astype(np.int64,copy=False)


def rotate_pose(positions,quarter_turns,k):
    """Rotate the whole canvas CCW by k quarter turns and normalize origin."""
    p,q=_pose(positions,quarter_turns);p=p.copy()
    for _ in range(int(k)%4):p=np.column_stack((-p[:,1],p[:,0]))
    p-=p.min(axis=0)
    return p,(q+int(k))%4


def compare_poses(positions,quarter_turns,reference_positions,reference_turns,
                  joint_boundary=.95,neighbor_boundary=.95):
    """Five pose scores under one common global C4; match pose-v1.0 tie rule.

    Prefer passing candidates, then joint pose score, oriented neighbor score,
    then smaller k. A reference is an explicit supplied array, never loaded.
    """
    p,q=_pose(positions,quarter_turns);ref,rq=_pose(reference_positions,reference_turns)
    if p.shape!=ref.shape:raise ValueError('piece counts differ')
    for bound in [joint_boundary,neighbor_boundary]:
        if type(bound) not in (int,float) or not np.isfinite(bound) or not 0<=bound<=1:
            raise ValueError('finite scalar boundary in [0,1] required')
    delta=ref[None]-ref[:,None]
    first,second=np.where((np.abs(delta).sum(axis=2)==1)&np.triu(np.ones((len(ref),len(ref)),bool),1))
    if not len(first):raise ValueError('neighbor metric undefined for no-seam reference')
    candidates=[]
    for k in range(4):
        a,b=rotate_pose(p,q,k)
        pc=np.all(a==ref,axis=1);qc=b==rq
        nc=np.all(a[second]-a[first]==ref[second]-ref[first],axis=1)
        metrics={'position_accuracy':float(pc.mean()),'orientation_accuracy':float(qc.mean()),
                 'joint_piece_accuracy':float((pc&qc).mean()),'neighbor_accuracy':float(nc.mean()),
                 'oriented_neighbor_accuracy':float((nc&qc[first]&qc[second]).mean())}
        passed=metrics['joint_piece_accuracy']>=joint_boundary and metrics['oriented_neighbor_accuracy']>=neighbor_boundary
        candidates.append({'global_quarter_turn':k,'passed':bool(passed),'metrics':metrics})
    chosen=max(candidates,key=lambda x:(x['passed'],x['metrics']['joint_piece_accuracy'],
                                        x['metrics']['oriented_neighbor_accuracy'],-x['global_quarter_turn']))
    k=chosen['global_quarter_turn'];a,b=rotate_pose(p,q,k)
    return {'global_quarter_turn':k,'passed':chosen['passed'],'metrics':chosen['metrics'],
            'candidates':candidates,'aligned_positions':a,'aligned_quarter_turns':b}


def render_tiles(tiles,positions,quarter_turns):
    """Assemble all original pixels exactly, preserving original tile dtype."""
    tiles=np.asarray(tiles)
    if tiles.ndim==5 and tiles.shape[0]==1:tiles=tiles[0]
    p,q=_pose(positions,quarter_turns)
    if tiles.ndim!=4 or len(tiles)!=len(p) or tiles.shape[1]!=tiles.shape[2]:
        raise ValueError('expected (N,P,P,C) square tile arrays')
    if not np.isfinite(tiles).all():raise ValueError('nonfinite tile values')
    side=tiles.shape[1];extent=p.max(axis=0)+1
    result=np.empty((int(extent[0])*side,int(extent[1])*side,tiles.shape[3]),tiles.dtype)
    for tile,(row,col),r in zip(tiles,p,q):
        result[row*side:(row+1)*side,col*side:(col+1)*side]=np.rot90(tile,int(r))
    return result


def dynamic_range_nrmse(image,reference):
    """RMS error / (max(reference)-min(reference)), all pixels/channels."""
    a=np.asarray(image);b=np.asarray(reference)
    if a.shape!=b.shape or not a.size or np.iscomplexobj(a) or np.iscomplexobj(b):
        raise ValueError('equal nonempty real image arrays required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite image')
    a=a.astype(np.float64);b=b.astype(np.float64);scale=float(np.max(b)-np.min(b))
    if scale<=0:raise ValueError('dynamic-range NRMSE undefined for constant reference')
    return float(np.sqrt(np.mean((a-b)**2))/scale)


def plot_reconstructions(tiles,source_pose,fresh_pose,truth_pose):
    """Return one comparison figure and metrics computed from supplied poses.

    source_pose/fresh_pose/truth_pose are (positions,quarter_turns) pairs.
    Source/fresh are separately aligned by one common C4 each to known truth.
    """
    import matplotlib.pyplot as plt
    source=compare_poses(*source_pose,*truth_pose);fresh=compare_poses(*fresh_pose,*truth_pose)
    reference=render_tiles(tiles,*truth_pose)
    source_image=render_tiles(tiles,source['aligned_positions'],source['aligned_quarter_turns'])
    fresh_image=render_tiles(tiles,fresh['aligned_positions'],fresh['aligned_quarter_turns'])
    source['dynamic_range_nrmse']=dynamic_range_nrmse(source_image,reference)
    fresh['dynamic_range_nrmse']=dynamic_range_nrmse(fresh_image,reference)
    if np.asarray(tiles).dtype.kind in 'iu':display_scale=float(np.iinfo(np.asarray(tiles).dtype).max)
    else:display_scale=1.
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for ax,image,title in zip(axes.flat[:3],[reference,source_image,fresh_image],
        ['Known acquisition picture','Original selected source run','Fresh local reconstruction']):
        ax.imshow(image.astype(float)/display_scale);ax.set_title(title);ax.axis('off')
    error=np.mean(abs(fresh_image.astype(float)-reference.astype(float)),axis=2)
    maximum=float(np.max(error))
    artist=axes[1,1].imshow(error,cmap='magma',vmin=0,vmax=maximum if maximum>0 else 1.)
    title='Fresh absolute RGB error (mean DN)' if maximum>0 else 'Fresh absolute RGB error (all pixels 0 DN)'
    axes[1,1].set_title(title);axes[1,1].axis('off')
    fig.colorbar(artist,ax=axes[1,1],shrink=.8,label='DN')
    return fig,{'source':source,'fresh':fresh}
